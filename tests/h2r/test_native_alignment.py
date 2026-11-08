"""Behavioral boundaries of the native recipe, independent of physical backend."""

import ast
from bisect import bisect
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from maniptrans_h2r.curriculum import ResetSchedule
from maniptrans_h2r.residual_reward import hand_reward, residual_reward
from maniptrans_h2r.robot_physics import require_contacts
from maniptrans_h2r.rsi import optimize_sequence
from test_reproduction import reward_fixture

ROOT = Path(__file__).resolve().parents[2]


def native_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def test_failure_grace_counts_calls_before_current_action():
    robot, state, target = reward_fixture()
    for body in state['bodies'].values():
        body[:, 0] += 1  # Bad tracking, with velocity sanity still valid.
    power, actions = torch.zeros(2, 7), torch.zeros(2, 14)
    history = {'right': torch.ones(2, 3, 5, dtype=torch.bool)}
    assert hand_reward(robot, 'right', state, target, power, actions, torch.tensor([19, 20]), .7)[1].tolist() == [False, True]
    assert residual_reward(robot, [0], state, target, power, actions, torch.tensor([7, 8]), .7, history)[1].tolist() == [False, True]


def test_coverage_is_required_even_when_contact_schema_matches():
    obs = dict(contacts=True, contact_schema='h2r-body-contacts-v1')
    with pytest.raises(ValueError, match='coverage'):
        require_contacts(dict(observation=obs))
    require_contacts(dict(observation=dict(obs, contact_coverage='robot-all-v1')))


def test_curriculum_matches_native_reset_method_and_property_helpers():
    # Execute unchanged native method/helpers without importing Isaac Gym. The
    # stub implements only property storage, never the schedule or sampling law.
    class SimParams:
        def __init__(self):
            self.gravity = SimpleNamespace(x=0., y=0., z=-9.81)
    props = [[SimpleNamespace(friction=f) for f in (.2, 1., 2.4)] for _ in range(3)]
    sim = SimParams()
    frame = [0]
    gym = SimpleNamespace(get_frame_count=lambda _: frame[0], get_sim_params=lambda _: sim,
        set_sim_params=lambda *args: None, find_actor_handle=lambda *args: 0)
    ns = dict(np=np, torch=torch, bisect=bisect, deepcopy=deepcopy, gymapi=SimpleNamespace(SimParams=SimParams),
        check_buckets=lambda *args: None,
        get_property_setter_map=lambda _: {'rigid_shape_properties': lambda *args: None},
        get_default_setter_args=lambda _: {'rigid_shape_properties': []},
        get_property_getter_map=lambda _: {'rigid_shape_properties': lambda env, handle: props[env]})
    native_functions(ROOT/'maniptrans_envs/lib/utils/dr_utils.py',
                     {'generate_random_samples', 'get_bucketed_val', 'apply_random_samples'}, ns)
    native_functions(ROOT/'maniptrans_envs/lib/envs/core/vec_task.py',
                     {'apply_randomizations', 'save_getattr', 'get_external_sample'}, ns)
    native = SimpleNamespace(gym=gym, sim=None, first_randomization=True, num_envs=3,
        envs=list(range(3)), last_rand_step=0, randomize_buf=torch.zeros(3, dtype=torch.long),
        reset_buf=torch.zeros(3, dtype=torch.bool), original_props={}, actor_params_generator=None,
        extern_actor_params=[None]*3, sim_initialized=True)
    recipe = yaml.safe_load((ROOT/'main/cfg/task/ResDexHand.yaml').read_text())['task']['randomization_params']
    recipe['actor_params'] = {'object': recipe['actor_params']['manip_obj']}
    adapted = ResetSchedule(3, [.2, 1., 2.4])
    events = {0: [0, 1, 2], 31: [0], 32: [1], 33: [0, 2], 64: [1],
              65: [0, 2], 1919: [0], 1920: [1, 2], 1952: [0, 1], 1984: [2]}
    for step in range(1985):
        frame[0] = step
        if step:
            adapted.advance()
            native.randomize_buf += 1
        if step in events:
            ids = events[step]
            native.reset_buf[:] = False
            native.reset_buf[ids] = True
            ns['apply_randomizations'](native, recipe)
            adapted.reset(ids, step)
            np.testing.assert_array_equal(adapted.age, native.randomize_buf)
            np.testing.assert_array_equal(adapted.friction, [[p.friction for p in row] for row in props])
            assert adapted.gravity_scale * -9.81 == sim.gravity.z
            assert adapted.last_step == native.last_rand_step
        if step == 65:
            restored = ResetSchedule(3, [.2, 1., 2.4])
            restored.load_state_dict(adapted.state_dict())
            adapted = restored
    assert adapted.friction[0, 0] == 1  # Native absolute bucket lower bound.


@pytest.mark.parametrize('chunk_size', [256, 513])
def test_ik_chunked_gradient_and_adam_match_full_sequence(chunk_size):
    torch.manual_seed(0)
    initial = torch.randn(513, 2, dtype=torch.float64)
    expected = torch.nn.Parameter(initial.clone())
    actual = torch.nn.Parameter(initial.clone())
    opt = torch.optim.Adam([actual], lr=.0004)
    oracle = torch.optim.Adam([expected], lr=.0004)
    for _ in range(12):
        oracle.zero_grad()
        expected.square().mean().backward()
        oracle.step()
    stats = optimize_sequence(opt, lambda a, b: actual[a:b].square().sum()/actual.numel(),
                              len(actual), 12, lambda: None, chunk_size)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-15)
    assert stats['iterations'] == 12 and stats['stop_reason'] == 'iteration_budget'


@pytest.mark.parametrize('improvement,expected', [(0., 200), (-.1, 200), (9e-6, 200), (11e-6, 300)])
def test_ik_native_stop_uses_global_100_iteration_checks(improvement, expected):
    p = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))
    opt = torch.optim.SGD([p], lr=0.)
    count = [0]
    def loss(a, b):
        if a == 0:
            count[0] += 1
        value = 1. if count[0] <= 100 else 1.-improvement
        return p*0+value*(b-a)/257
    stats = optimize_sequence(opt, loss, 257, 300, lambda: None)
    assert stats['iterations'] == expected
    assert stats['stop_reason'] == 'native_loss_improvement'


@pytest.mark.parametrize('sides', [['right'], ['right', 'left']])
def test_native_forward_and_ppo_updates_with_resampled_bases(tmp_path, sides):
    import os
    from types import MethodType
    from rl_games.algos_torch import torch_ext
    from rl_games.common import common_losses
    from rl_games.common.schedulers import AdaptiveScheduler
    from maniptrans_h2r.policy import build, parameters
    from maniptrans_h2r.training import ReproductionAgent
    source = os.environ.get('H2R_NATIVE_RL_SOURCE')
    if not source:
        pytest.skip('set H2R_NATIVE_RL_SOURCE to the unmodified upstream lib/rl directory')
    source = Path(source)
    torch.set_num_threads(2)
    dims = dict(privileged=4, proprioception=5, target=6)
    base = build(parameters(dims, dims, ['right'], {}, 8, True), dims, 8, 'cpu')
    path = tmp_path/'base.pth'
    torch.save({'model': base.state_dict()}, path)
    full = {k: n*len(sides) for k, n in dims.items()}
    params = parameters(full, dims, sides, {s: str(path) for s in sides}, 8*len(sides), True)
    model = build(params, full, 8*len(sides), 'cpu')
    oracle = deepcopy(model)
    suffix = 'sh' if len(sides) == 1 else 'bih'
    forward = native_functions(source/f'network_builder_residual_{suffix}.py', {'forward'}, {'torch': torch})['forward']
    oracle.a2c_network.forward = MethodType(forward, oracle.a2c_network)
    native_calc = native_functions(source/'agent.py', {'calc_gradients'},
                                   dict(torch=torch, common_losses=common_losses, torch_ext=torch_ext))['calc_gradients']
    observations = {k: torch.randn(4, n) for k, n in full.items()}
    model.eval(); oracle.eval()
    with torch.no_grad():
        torch.manual_seed(13)
        rollout = model(dict(is_train=False, obs=deepcopy(observations)))
        rng = torch.get_rng_state()
        torch.manual_seed(13)
        expected = oracle(dict(is_train=False, obs=deepcopy(observations)))
        assert torch.equal(torch.get_rng_state(), rng)
        for key, value in rollout.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    batch = dict(old_values=rollout['values'], old_logp_actions=rollout['neglogpacs'],
        advantages=torch.randn(4), mu=rollout['mus'], sigma=rollout['sigmas'],
        returns=torch.randn(4, 1), actions=rollout['actions'], obs=observations)
    agents = []
    for network in (model, oracle):
        agent = ReproductionAgent.__new__(ReproductionAgent)
        agent.__dict__.update(model=network.train(), e_clip=.2, is_rnn=False, mixed_precision=False,
            actor_loss_func=common_losses.actor_loss, ppo=True, has_value_loss=True, clip_value=True,
            bound_loss_type='bound', critic_coef=4., entropy_coef=0., bounds_loss_coef=.0001,
            multi_gpu=False, truncate_grads=True, grad_norm=1., last_lr=.0005, ppo_device='cpu',
            config=dict(error_if_nonfinite_gradients=True),
            diagnostics=SimpleNamespace(mini_batch=lambda *args: None),
            scaler=torch.amp.GradScaler('cpu', enabled=False),
            optimizer=torch.optim.Adam((p for p in network.parameters() if p.requires_grad), lr=.0005, eps=1e-8))
        agents.append(agent)
    agents[1].calc_gradients = MethodType(native_calc, agents[1])
    schedulers = [AdaptiveScheduler(.008) for _ in agents]
    samples = [[], []]
    for i, network in enumerate((model, oracle)):
        network.a2c_network.register_forward_hook(lambda module, args, out, i=i: samples[i].append(out[-1].detach().clone()))
    for step in range(4):
        for i, agent in enumerate(agents):
            torch.manual_seed(20+step)
            agent.calc_gradients(deepcopy(batch))
            lr, _ = schedulers[i].update(agent.last_lr, 0., 1, 0, float(agent.train_result[3]))
            agent.update_lr(lr)
        for actual, expected in zip(agents[0].train_result, agents[1].train_result):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert agents[0].last_lr == agents[1].last_lr
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, oracle.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(samples[0][-1], samples[1][-1], rtol=0, atol=0)
    assert not torch.equal(samples[0][0], samples[0][1])
