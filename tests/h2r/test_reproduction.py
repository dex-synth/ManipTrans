from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from lib.rl.rewards import manipulation_reward
from maniptrans_h2r.hand_reference import continuous_euler, quaternion_signs
from maniptrans_h2r.motion import MotionCache, pack
from maniptrans_h2r.policy import build, parameters
from maniptrans_h2r.residual_reward import hand_inputs, native_call, residual_reward
from maniptrans_h2r.robot_physics import require_clock
from maniptrans_h2r.rsi import sampling_probabilities
from maniptrans_h2r.control import PositionControl


def test_euler_branches_are_continuous_through_wrap_and_gimbal():
    angles = np.array([[.1, .2, 3.13], [.1, .2, 3.15], [.1, np.pi/2, 3.15]])
    matrices = Rotation.from_euler('XYZ', angles).as_matrix()
    out = continuous_euler(matrices, np.full(3, -6.2), np.full(3, 6.2), previous=angles[0])
    np.testing.assert_allclose(Rotation.from_euler('XYZ', out).as_matrix(), matrices, atol=1e-6)
    assert abs(out[1, 2]-out[0, 2]) < .1
    with pytest.raises(ValueError, match='branch'):
        continuous_euler(matrices, np.full(3, -.01), np.full(3, .01))


def test_quaternion_signs_and_clock_do_not_silently_change_semantics():
    q = quaternion_signs(np.array([[-1., 0, 0, 0], [1., 0, 0, 0]]))
    np.testing.assert_array_equal(q, [[1, 0, 0, 0], [1, 0, 0, 0]])
    require_clock(dict(dt_s=1/120, control_ticks=2))
    with pytest.raises(ValueError):
        require_clock(dict(dt_s=1/240, control_ticks=8))


def test_motion_cache_visits_more_windows_than_worlds_and_clamps_only_queries(tmp_path):
    data = np.arange(500*145, dtype=np.float32).reshape(500, 145)
    np.save(tmp_path/'motion.npy', data)
    manifest = dict(synthetic_fraction=0, windows=[dict(offset=i*100, length=100, dataset='oakink2', source_side='right', synthetic=False) for i in range(5)])
    cache = MotionCache(tmp_path, manifest, 2, 'right', 'cpu', allow_partial=True)
    cache.reset(np.array([0, 1]), fixed_windows=[3, 4], from_start=True)
    np.testing.assert_array_equal(cache.sample()['right_human'][0].numpy().reshape(-1), data[300, :63])
    for _ in range(99):
        cache.advance()
    np.testing.assert_array_equal(cache.sample(1)['right_human'][1].numpy().reshape(-1), data[499, :63])
    assert cache.transitions.tolist() == [0, 0, 0, 99, 99]
    for _ in range(100):
        cache.reset(np.array([0, 1]))
    assert np.all(cache.visits > 0)
    with pytest.raises(ValueError, match='sampling pools'):
        MotionCache(tmp_path, manifest, 2, 'right', 'cpu')


def test_rsi_probabilities_preserve_original_indices_and_partial_final_bin():
    valid = np.array([False, True, False, True, True])
    p = sampling_probabilities(5, valid)
    np.testing.assert_allclose(p, [0, 1/2.9, 0, 1.9/2.9, 0])
    with pytest.raises(ValueError, match='no_valid_rsi'):
        sampling_probabilities(5, np.zeros(5, bool))


def test_v2_control_clips_enabled_ranges_and_fingers_only():
    robot = dict(joint_names=list(range(7)), lower=[-1]*7, upper=[1]*7,
                 command_lower=[-1e20]*6+[-.3], command_upper=[1e20]*5+[.2, .3], hands={'right': {'indices': list(range(7))}})
    control = PositionControl(robot, 1, 'cpu', dict(schema='virtual-wrist-position-v2', wrist_translation_m=.05, wrist_rotation_rad=.3, moving_average=1))
    action = torch.ones(1, 14)
    output = control.decode(action, torch.ones(1, 7), ['right'], True)
    assert output[0, 0] > 1  # Unbounded wrist command is not clipped to q range.
    assert output[0, 5].item() == pytest.approx(.2)
    assert output[0, 6].item() == pytest.approx(.3)


def reward_fixture(parts=1):
    robot = dict(sides=['right'], hands={'right': dict(indices=list(range(7)), bodies=[f'b{i}' for i in range(21)])})
    bodies = torch.zeros(2, 21, 13); bodies[..., 3] = 1
    objects = torch.zeros(2, parts, 13); objects[..., 3] = 1
    state = dict(robot_q=torch.zeros(2, 7), robot_dq=torch.zeros(2, 7),
        bodies={f'b{i}': bodies[:, i] for i in range(21)}, objects=objects, tip_force={'right': torch.ones(2, 5, 3)})
    target = dict(right_wrist=bodies[:, 0].clone(), right_human=bodies[:, :, :3].clone(),
                  right_human_velocity=bodies[:, :, 7:10].clone(), object_state=objects.clone(), right_tip_distance=torch.zeros(2, 5))
    return robot, state, target


def test_all_part_aggregation_equals_native_single_object_and_preserves_total_weight():
    robot, state, target = reward_fixture()
    power, actions, elapsed = torch.zeros(2, 7), torch.zeros(2, 14), torch.full((2,), 20, dtype=torch.long)
    history = {'right': torch.ones(2, 3, 5, dtype=torch.bool)}
    actual, goal = hand_inputs(robot, 'right', state, target, power)
    for values in (actual, goal):
        values.update(manip_obj_pos=torch.zeros(2, 3), manip_obj_quat=torch.tensor([[0., 0, 0, 1]]).repeat(2, 1), manip_obj_vel=torch.zeros(2, 3), manip_obj_ang_vel=torch.zeros(2, 3))
    goal.update(tip_force=state['tip_force']['right'], tips_distance=target['right_tip_distance'], tip_contact_state=history['right'])
    native = native_call(manipulation_reward, actual, goal, actions, elapsed, .7)
    result = residual_reward(robot, [0], state, target, power, actions, elapsed, .7, history)
    torch.testing.assert_close(result[0], native[0])
    torch.testing.assert_close(result[1], native[3])
    robot, state, target = reward_fixture(3)
    multi = residual_reward(robot, [0, 1, 2], state, target, power, actions, elapsed, .7, history)
    torch.testing.assert_close(multi[0], result[0])
    target['object_state'][:, 2, 0] = .1
    assert residual_reward(robot, [0, 1, 2], state, target, power, actions, elapsed, .7, history)[1].all()


def test_bimanual_given_base_sample_does_not_consume_rng_and_freezes_norms(tmp_path):
    torch.set_num_threads(2)
    dims = dict(privileged=4, proprioception=5, target=6)
    base = build(parameters(dims, dims, ['right'], {}, 8, True), dims, 8, 'cpu')
    path = tmp_path/'base.pth'; torch.save({'model': base.state_dict()}, path)
    full = {k: n*2 for k, n in dims.items()}
    params = parameters(full, dims, ['right', 'left'], dict(right=str(path), left=str(path)), 16, True)
    model = build(params, full, 16, 'cpu').train()
    frozen = {k: v.clone() for k, v in model.state_dict().items() if 'base_model' in k}
    obs = {k: torch.randn(3, n) for k, n in full.items()}
    samples = torch.randn(3, 16)
    before = torch.get_rng_state().clone()
    model(dict(is_train=True, prev_actions=torch.zeros(3, 16), obs=obs, base_actions=samples))
    assert torch.equal(torch.get_rng_state(), before)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in frozen.items())
