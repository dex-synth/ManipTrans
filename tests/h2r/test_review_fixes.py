"""Regressions for wrist branches, physical power and silent numerical failures."""

from types import SimpleNamespace
import json
import random

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from maniptrans_h2r.hand_reference import continuous_euler
from maniptrans_h2r.policy import build, parameters
from maniptrans_h2r.robot_physics import require_compatibility, signature


def test_wrist_chooses_one_branch_for_the_whole_clip():
    angles = np.column_stack([np.full(601, .1), np.full(601, .2), np.linspace(3, 9, 601)])
    rotations = Rotation.from_euler('XYZ', angles).as_matrix()
    q = continuous_euler(rotations, np.full(3, -6.2), np.full(3, 6.2))
    assert np.abs(np.diff(q, axis=0)).max() < .011
    assert np.abs(q).max() <= 6.2
    np.testing.assert_allclose(Rotation.from_euler('XYZ', q).as_matrix(), rotations, atol=1e-6)


def test_wrist_rejects_a_clip_that_cannot_fit_without_wrapping():
    angles = np.column_stack([np.full(1500, .1), np.full(1500, .2), np.linspace(0, 4*np.pi, 1500)])
    with pytest.raises(ValueError, match='continuous wrist rotation branch'):
        continuous_euler(Rotation.from_euler('XYZ', angles).as_matrix(), np.full(3, -6.2), np.full(3, 6.2))


def test_rsi_preserves_reference_branch_and_rejects_noise_across_limit():
    reference = np.array([-.871466, .9995453, 3.7817035])
    q = continuous_euler(Rotation.from_euler('XYZ', reference[None]).as_matrix(),
                         np.full(3, -6.2), np.full(3, 6.2), previous=reference)
    np.testing.assert_allclose(q[0], reference, atol=1e-6)
    with pytest.raises(ValueError, match='anchored'):
        continuous_euler(Rotation.from_euler('XYZ', [[.1, .2, 6.22]]).as_matrix(),
                         np.full(3, -6.2), np.full(3, 6.2), previous=[.1, .2, 6.19])


def test_power_uses_post_step_velocity_and_preserves_short_terminal(monkeypatch):
    from maniptrans_h2r import mjlab_env
    views = dict(qvel=torch.ones(2, 1), qpos=torch.zeros(2, 1), ctrl=torch.zeros(2, 1),
                 qfrc_actuator=torch.zeros(2, 1))
    def simulate():
        views['qvel'] += 1
        views['qfrc_actuator'][:] = 2
    env = SimpleNamespace(count=2, device='cpu', done=torch.zeros(2, dtype=torch.bool),
        views=views, robot=dict(actuator_indices=[0], ctrl_joint_indices=[0], qvel_indices=[0]),
        control=SimpleNamespace(decode=lambda *args: torch.zeros(2, 1)), stage='residual', sides=[],
        sim=SimpleNamespace(step=simulate), data=SimpleNamespace(overflow=torch.zeros(2)),
        curriculum=None, completed_controls=torch.zeros(2, dtype=torch.long),
        tick=torch.zeros(2, dtype=torch.long), elapsed_ticks=torch.zeros(2, dtype=torch.long), common_steps=0,
        scale=.7, contacts=lambda: {}, state=lambda: {}, target=lambda *args: dict(robot_q=torch.zeros(2, 1)),
        advance_reference=lambda: None, features=lambda *args: {},
        reward=lambda state, target, power, actions: (power[:, 0], torch.zeros(2, dtype=torch.bool), {}))
    env.remaining_ticks = lambda: torch.tensor([1, 2])-env.tick
    monkeypatch.setattr(mjlab_env.wp, 'to_torch', lambda value: value)
    _, reward, done, _ = mjlab_env.MjlabVecEnv.step(env, torch.zeros(2, 2))
    torch.testing.assert_close(reward, torch.tensor([4., 6.]))
    torch.testing.assert_close(views['qvel'][:, 0], torch.tensor([2., 3.]))
    assert done.all() and env.tick.tolist() == [1, 2]


def physics_fixture(extra=''):
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option timestep="0.008333333333333333"/>
      <worldbody><geom name="floor" type="plane" size="1 1 .1"/>
        <body name="hand"><joint name="wrist" range="-6.2 6.2"/>
          <geom name="hand_geom" size=".1"/></body></worldbody>
      <actuator><position name="servo" joint="wrist" kp="10"/></actuator>{extra}</mujoco>''')
    robot = dict(hands={'right': dict(joints=['wrist'])}, actuator_indices=[0])
    return model, robot


@pytest.mark.parametrize('field,value', [('jnt_solref', [.002, .5]), ('jnt_solimp', [.8, .9, .01, .5, 2]),
                                        ('dof_solref', [.001, .5]), ('jnt_margin', .01)])
def test_signature_includes_joint_constraint_physics(field, value):
    model, robot = physics_fixture()
    before = signature(model, robot, 'right')
    getattr(model, field)[0] = value
    assert signature(model, robot, 'right')['robot_physics_id'] != before['robot_physics_id']


def test_signature_covers_ccd_and_rejects_unsigned_hand_constraints():
    model, robot = physics_fixture()
    before = signature(model, robot, 'right')
    require_compatibility(before)
    model.opt.ccd_iterations = 2
    assert signature(model, robot, 'right')['physics_options_id'] != before['physics_options_id']
    with pytest.raises(ValueError, match='Re-prepare'):
        require_compatibility(dict(reference_schema='hand-reference-v2'))


@pytest.mark.parametrize('constraint', [
    '<contact><pair geom1="floor" geom2="hand_geom"/></contact>',
    '<equality><joint joint1="wrist"/></equality>',
    '<tendon><fixed name="coupling"><joint joint="wrist" coef="1"/></fixed></tendon>',
])
def test_unsigned_hand_constraints_fail_explicitly(constraint):
    model, robot = physics_fixture(constraint)
    with pytest.raises(ValueError, match='Unsupported.*hand'):
        signature(model, robot, 'right')


@pytest.mark.parametrize('sides', [[], ['right'], ['right', 'left']])
@pytest.mark.parametrize('head,value', [('mu', float('nan')), ('value', float('inf')), ('sigma', 1000.)])
def test_h2r_policy_fails_before_nan_to_num(tmp_path, sides, head, value):
    torch.set_num_threads(2)
    dims = dict(privileged=4, proprioception=5, target=6)
    base = build(parameters(dims, dims, ['right'], {}, 8, True), dims, 8, 'cpu')
    path = tmp_path/'base.pth'
    torch.save(dict(model=base.state_dict()), path)
    count = max(1, len(sides))
    full = {k: v*count for k, v in dims.items()}
    model = build(parameters(full, dims, sides or ['right'], {s: str(path) for s in sides},
                             8*count, True), full, 8*count, 'cpu').eval()
    with torch.no_grad():
        tensor = getattr(model.a2c_network, head)
        (tensor if head == 'sigma' else tensor.weight).fill_(value)
    with pytest.raises(FloatingPointError, match='policy output'):
        model(dict(is_train=False, obs={k: torch.zeros(1, v) for k, v in full.items()},
                   base_noise=torch.zeros(1, 8*count)))


def test_nonfinite_gradient_is_rejected_before_optimizer_step():
    from maniptrans_h2r.training import ReproductionAgent
    agent = ReproductionAgent.__new__(ReproductionAgent)
    agent.model = torch.nn.Linear(1, 1)
    agent.model.weight.grad = torch.full_like(agent.model.weight, float('nan'))
    before = agent.model.weight.detach().clone()
    with pytest.raises(FloatingPointError, match='gradients'):
        agent.trancate_gradients_and_step()
    torch.testing.assert_close(agent.model.weight, before)


def test_overflowing_gradient_norm_does_not_silently_zero_the_update():
    from maniptrans_h2r.training import ReproductionAgent
    agent = ReproductionAgent.__new__(ReproductionAgent)
    agent.model = torch.nn.Linear(2, 1)
    agent.model.weight.grad = torch.full_like(agent.model.weight, 1e30)
    agent.multi_gpu, agent.truncate_grads, agent.grad_norm = False, True, 1.
    agent.config = dict(error_if_nonfinite_gradients=True)
    agent.optimizer = object()
    agent.scaler = SimpleNamespace(unscale_=lambda optimizer: None)
    with pytest.raises(RuntimeError, match='non-finite'):
        agent.trancate_gradients_and_step()
    assert (agent.model.weight.grad == 1e30).all()


def test_diagnostics_run_at_epoch_boundaries_without_changing_rng_or_mode(monkeypatch):
    from maniptrans_h2r.training import ReproductionAgent
    agent = ReproductionAgent.__new__(ReproductionAgent)
    agent.model = torch.nn.Linear(1, 1).train()
    agent.recipe = dict(save_every=100, diagnostic_every=2)
    agent.epoch_num, agent.last_diagnostic = 0, -1
    agent.last_final_diagnostic = -1
    seen = []
    def callback(model, epoch, *, mean, final):
        seen.append(epoch)
        model.eval()
        random.random(); np.random.random(); torch.rand(3)
    agent.diagnostic_callback = callback
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [])
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', lambda values: None)
    random.seed(123); np.random.seed(123); torch.manual_seed(123)
    expected = (random.random(), np.random.random(), torch.rand(3))
    random.seed(123); np.random.seed(123); torch.manual_seed(123)
    for _ in range(3):
        agent.update_epoch()
    agent.diagnostic()  # Final epoch is always diagnosed, even off-period.
    agent.diagnostic()  # Do not repeat the same completed epoch.
    assert seen == [0, 2, 3] and agent.model.training
    assert (random.random(), np.random.random()) == expected[:2]
    torch.testing.assert_close(torch.rand(3), expected[2])


def test_residual_diagnostic_records_rejected_initialization(tmp_path):
    from maniptrans_h2r.diagnostic import residual_tracking
    (tmp_path/'initialization.json').write_text(json.dumps(dict(valid=False, collisions=[dict(depth_m=.01)])))
    report = residual_tracking(None, tmp_path, 'cpu')
    assert report['status'] == 'initialization_failed'
    assert report['coverage'] == 0 and not report['success']
