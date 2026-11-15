"""Measured-state wrist control across training, deployment and old artifacts."""

from types import SimpleNamespace

import pytest
import torch

from maniptrans_h2r.clock import DECIMATION
from maniptrans_h2r.control import CONTROL, PositionControl
from maniptrans_h2r.evaluate import Controller


@pytest.fixture
def robot():
    return dict(joint_names=[f'q{i}' for i in range(7)], lower=[-5.]*3+[-6.2]*3+[-1.],
                upper=[5.]*3+[6.2]*3+[1.], command_lower=[-5.]*3+[-6.2]*3+[-.8],
                command_upper=[5.]*3+[6.2]*3+[.8], sides=['right'],
                hands={'right': dict(indices=list(range(7)), bodies=['hand'], contact_bodies=['hand'])},
                qpos_indices=list(range(7)), qvel_indices=list(range(7)),
                actuator_indices=list(range(7)), ctrl_joint_indices=list(range(7)))


def test_current_position_not_reference_or_previous_target(robot):
    control = PositionControl(robot, 1, 'cpu', CONTROL)
    actual = torch.tensor([[.2, .3, .4, 3.2, -.2, .5, -.4]])
    action = torch.full((1, 7), .5)
    expected = actual.clone()
    expected[:, :3] += .06
    expected[:, 3:6] += .5
    expected[:, 6] = .5
    torch.testing.assert_close(control.decode(action, ['right'], False, actual), expected)
    actual[:, 0] -= .1  # A disturbance must change the next command, without integrating target history.
    expected[:, 0] -= .1
    torch.testing.assert_close(control.decode(action, ['right'], False, actual), expected)
    zero = control.decode(torch.zeros_like(action), ['right'], False, actual)
    torch.testing.assert_close(zero[:, :6], actual[:, :6])


def test_residual_composition_limits_and_no_angle_wrapping(robot):
    control = PositionControl(robot, 1, 'cpu', CONTROL)
    actual = torch.tensor([[4.99, 0., 0., 6.19, 0., 0., 0.]])
    action = torch.ones(1, 14)
    result = control.decode(action, ['right'], True, actual)
    torch.testing.assert_close(result, torch.tensor([[5., .36, .36, 6.2, 3., 3., .8]]))


def test_bimanual_residual_uses_each_hands_measured_origin(robot):
    robot = dict(robot, joint_names=robot['joint_names']+[f'left_{n}' for n in robot['joint_names']],
                 hands=dict(right={'indices': list(range(7))}, left={'indices': list(range(7, 14))}))
    for key in ('lower', 'upper', 'command_lower', 'command_upper'):
        robot[key] = robot[key]*2
    control = PositionControl(robot, 1, 'cpu', CONTROL)
    actual = torch.tensor([[.2]*7+[-.3]*7])
    actions = torch.zeros(1, 28)
    actions[:, 7:14] = .5  # Left base only.
    actions[:, 14:21] = -.25  # Right residual only, with native factor 2.
    result = control.decode(actions, ['right', 'left'], True, actual)
    torch.testing.assert_close(result, torch.tensor([[.14]*3+[-.3]*3+[-.5]+[-.24]*3+[.2]*3+[.5]]))


@pytest.mark.parametrize('positions', [None, torch.zeros(1, 6), torch.full((1, 7), float('nan')),
                                      torch.full((1, 7), float('inf'))])
def test_current_position_is_required_and_finite(robot, positions):
    control = PositionControl(robot, 1, 'cpu', CONTROL)
    with pytest.raises(ValueError, match='measured joint positions'):
        control.decode(torch.zeros(1, 7), ['right'], False, positions)


def test_control_rejects_smoothing_and_other_versions(robot):
    for config in (dict(CONTROL, moving_average=.5), dict(CONTROL, schema='unknown'),
                   dict(CONTROL, schema='virtual-wrist-position-v2'), {k: v for k, v in CONTROL.items() if k != 'schema'}):
        with pytest.raises(ValueError):
            PositionControl(robot, 1, 'cpu', config)


def test_training_samples_measured_position_once_per_control(robot, monkeypatch):
    from maniptrans_h2r import mjlab_env
    views = dict(qpos=torch.full((1, 7), .25), qvel=torch.zeros(1, 7), ctrl=torch.zeros(1, 7),
                 qfrc_actuator=torch.zeros(1, 7))
    commands = []
    def simulate():
        commands.append(views['ctrl'].clone())
        views['qpos'] += .01
    env = SimpleNamespace(count=1, device='cpu', done=torch.zeros(1, dtype=torch.bool), views=views, robot=robot,
        sides=['right'], stage='imitator', control=PositionControl(robot, 1, 'cpu', CONTROL),
        sim=SimpleNamespace(step=simulate), data=SimpleNamespace(overflow=torch.zeros(1)),
        remaining_ticks=lambda: torch.tensor([DECIMATION]), target=lambda *args: dict(robot_q=torch.full((1, 7), -2.)),
        tick=torch.zeros(1, dtype=torch.long), elapsed_ticks=torch.zeros(1, dtype=torch.long), common_steps=0,
        completed_controls=torch.zeros(1, dtype=torch.long), curriculum=None, contacts=lambda: {'right': torch.zeros(1, 5, 3)},
        tip_force={'right': torch.zeros(1, 5, 3)}, contact_history={'right': torch.zeros(1, 3, 5, dtype=torch.bool)}, scale=1.,
        advance_reference=lambda: None, state=lambda: {}, features=lambda *args: {},
        reward=lambda *args: (torch.zeros(1), torch.zeros(1, dtype=torch.bool), {}))
    monkeypatch.setattr(mjlab_env.wp, 'to_torch', lambda value: value)
    mjlab_env.MjlabVecEnv.step(env, torch.ones(1, 7))
    assert len(commands) == DECIMATION == 2
    torch.testing.assert_close(commands[0], torch.tensor([[.37]*3+[1.25]*3+[.8]]))
    torch.testing.assert_close(commands[1], commands[0])
    mjlab_env.MjlabVecEnv.step(env, torch.zeros(1, 7))
    torch.testing.assert_close(commands[2][:, :6], torch.full((1, 6), .27))


def test_deployment_uses_current_wire_positions_for_each_episode(robot):
    client = Controller.__new__(Controller)
    client.robot, client.device = robot, 'cpu'
    client.meta = dict(object_bodies=['object'])
    client.orders = dict(robot=list(range(7)), object=[])
    client.ticks, client.dt_s, client.end_tick = 5, 1/300, 50  # A 1/60 s policy on 300 Hz evaluation physics.
    sampled = []
    client.reference = SimpleNamespace(sample=lambda t: sampled.append(t.tolist()) or {})
    client.features = lambda state, *args: state
    client.policy = lambda obs, ids: torch.zeros(len(ids), 14)
    client.controls = {name: PositionControl(robot, 1, 'cpu', CONTROL) for name in ('a', 'b')}
    client.next_ticks = dict(a=0, b=0)
    frames = [dict(episode_id=name, done=False, tick=0, robot_q=[value]*7, robot_dq=[0.]*7,
                   object_q=[], object_dq=[], bodies={'hand': [0.]*13, 'object': [0.]*13},
                   contacts=dict(solve_tick=None, records=[])) for name, value in [('a', .2), ('b', -.3)]]
    for tick in (0, 5):
        result = client.step(dict(type='frames', done=False, episodes=frames))['commands']
        for frame in frames:
            targets = result[frame['episode_id']]['joint_targets']
            assert [targets[n] for n in robot['joint_names'][:6]] == pytest.approx(frame['robot_q'][:6])
            frame['tick'] += 5
            frame['contacts']['solve_tick'] = frame['tick']-1
            frame['robot_q'] = [q+.1 for q in frame['robot_q']]
    # The reference is tracked one control period ahead of the observation, in seconds.
    assert sampled == [pytest.approx([1/60]*2), pytest.approx([2/60]*2)]
    with pytest.raises(ValueError, match='Unexpected control tick'):
        client.step(dict(type='frames', done=False, episodes=[dict(frames[0], tick=12)]))


def test_standalone_diagnostic_rejects_an_old_control_before_loading_model(monkeypatch):
    from maniptrans_h2r import diagnostic
    monkeypatch.setattr(diagnostic, 'load_artifact', lambda *args: (None, {'control': {'schema': 'virtual-wrist-position-v2'}}))
    with pytest.raises(ValueError, match='wrist control differs'):
        diagnostic.check_imitator(SimpleNamespace(imitator='old'))
