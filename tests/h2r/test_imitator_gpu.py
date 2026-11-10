"""Opt-in physical regression using an existing prepared imitator corpus."""

import gc
import os
from pathlib import Path

import pytest
import torch


def test_repeated_diagnostic_worlds_preserve_the_training_stream():
    prepared = os.environ.get('H2R_IMITATOR_PREPARED')
    if not prepared:
        pytest.skip('set H2R_IMITATOR_PREPARED to run the CUDA environment lifecycle check')
    from maniptrans_h2r.imitator_env import ImitatorEnv

    env = ImitatorEnv(Path(prepared), 2, 'cuda:0')
    actions = torch.zeros(2, env.action_dim, device='cuda:0')
    for _ in range(4):
        diagnostic = ImitatorEnv(Path(prepared), 2, 'cuda:0')
        assert diagnostic.warp_stream is env.warp_stream
        diagnostic.step(actions)
        torch.cuda.synchronize()
        del diagnostic
        gc.collect()
        env.reset_done()
        positions = env.state()['robot_q'].clone()
        env.step(actions)
        targets = env.views['ctrl'][:, env.robot['actuator_indices']]
        torch.testing.assert_close(targets[:, :6], positions[:, :6], rtol=0, atol=0)
        assert torch.isfinite(env.views['qpos']).all()
    torch.cuda.synchronize()


def test_physical_constraint_overflow_reports_the_limit_and_expansion_clears_it():
    if not os.environ.get('H2R_IMITATOR_PREPARED'):
        pytest.skip('set H2R_IMITATOR_PREPARED to enable CUDA physics checks')
    import mujoco
    import warp as wp
    from mjlab.sim import Simulation
    from maniptrans_h2r.buffers import check_overflow, SimulationOverflow
    from maniptrans_h2r.mjlab_env import simulation_config

    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <worldbody><geom type="plane" size="2 2 .1"/>
        <body pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
      </worldbody></mujoco>''')
    for limit in (1, 32):
        cfg = simulation_config(model, 'imitator', dict(contact_capacity=32, constraint_capacity=limit))
        sim = Simulation(num_envs=2, cfg=cfg, model=model, device='cuda:0')
        wp.to_torch(sim.wp_data.qpos)[:, 2] = .05
        torch.cuda.synchronize()
        sim.step()  # MJWarp marks NEFC in the end-of-step overflow check.
        if limit == 1:
            with pytest.raises(SimulationOverflow) as caught:
                check_overflow(sim.wp_data, 'step')
            assert caught.value.report['flags']['NEFC'] == 2
            assert min(caught.value.report['nefc']) > limit
        else:
            check_overflow(sim.wp_data, 'step')
            assert int(wp.to_torch(sim.wp_data.nefc).max()) > 1
        del sim
        gc.collect()


@pytest.mark.parametrize('fault_substep', [0, 1])
def test_epa_skip_preserves_other_worlds_and_resets_only_completed_episodes(tmp_path, monkeypatch, fault_substep):
    prepared = os.environ.get('H2R_IMITATOR_PREPARED')
    if not prepared:
        pytest.skip('set H2R_IMITATOR_PREPARED to enable CUDA recovery checks')
    import warp as wp
    from maniptrans_h2r.imitator_env import ImitatorEnv
    from maniptrans_h2r.buffers import OverflowType
    from maniptrans_h2r.recovery import EpaRecovery, STATE_KEYS

    env = ImitatorEnv(Path(prepared), 4, 'cuda:0', seed=42, epa_recovery=EpaRecovery(tmp_path))
    actions = torch.zeros(4, env.action_dim, device='cuda:0')
    healthy = torch.tensor([0, 2, 3], device='cuda:0')
    start_ticks = env.tick.clone()
    expected = {}
    step = env.sim.step
    calls = 0
    def inject():
        nonlocal calls
        if calls == 1:
            # Compare the same simulation at the substep boundary. Independent
            # GPU solves can differ slightly through parallel contact ordering.
            for key, value in expected.items():
                torch.testing.assert_close(env.views[key][healthy], value, rtol=0, atol=0)
        step()
        expected.update({key:env.views[key][healthy].clone() for key in STATE_KEYS})
        if calls == fault_substep:
            wp.to_torch(env.data.overflow)[1] |= int(OverflowType.EPA_HORIZON)
            # A corrupt row must never enter the next physics substep or observation.
            env.views['qpos'][1] = 1e6
            env.views['qvel'][1] = 1e6
        calls += 1
    monkeypatch.setattr(env.sim, 'step', inject)
    obs, reward, done, info = env.step(actions)
    assert calls == 2 and done[1] and reward[1] == 0
    assert info['error_masks'].tolist() == [1, 0, 1, 1]
    assert info['simulation_errors'].tolist() == [False, True, False, False]
    assert not env.success_buf[1] and not env.failure_buf[1]
    assert env.epa_recovery.skipped_episodes == 1
    for key, value in expected.items():
        torch.testing.assert_close(env.views[key][healthy], value, rtol=0, atol=0)
    for key in obs:
        assert torch.isfinite(obs[key]).all()
    torch.testing.assert_close(env.tick, start_ticks+2, rtol=0, atol=0)
    continuing = ~done
    before = {key:value[continuing].clone() for key, value in env.views.items()}
    env.reset_done()
    assert not env.done.any()
    for key, value in before.items():
        torch.testing.assert_close(env.views[key][continuing], value, rtol=0, atol=0)
    _, _, _, info = env.step(actions)
    assert not info['simulation_errors'].any()
    assert not wp.to_torch(env.data.overflow).any()


def test_diagnostic_retains_epa_failed_windows_in_denominator(tmp_path, monkeypatch):
    prepared = os.environ.get('H2R_IMITATOR_PREPARED')
    if not prepared:
        pytest.skip('set H2R_IMITATOR_PREPARED to enable CUDA recovery checks')
    import warp as wp
    from maniptrans_h2r import diagnostic, imitator_env
    from maniptrans_h2r.buffers import OverflowType
    from maniptrans_h2r.io import read_json
    from maniptrans_h2r.recovery import EpaRecovery

    path = Path(prepared)
    windows = read_json(path/'motion_manifest.json')['windows']
    selected = sorted(range(len(windows)), key=lambda i: windows[i]['length'])[:2]
    monkeypatch.setattr(diagnostic, 'read_json', lambda _:dict(windows=selected, overlap='test'))
    env_class = imitator_env.ImitatorEnv
    def make_env(*args, **kwargs):
        env = env_class(*args, **kwargs)
        original = env.sim.step
        calls = 0
        def inject():
            nonlocal calls
            original()
            if calls == 0:
                wp.to_torch(env.data.overflow)[1] |= int(OverflowType.EPA_HORIZON)
            calls += 1
        monkeypatch.setattr(env.sim, 'step', inject)
        return env
    monkeypatch.setattr(imitator_env, 'ImitatorEnv', make_env)
    class ZeroPolicy:
        def eval(self):
            pass
        def __call__(self, inputs):
            return {'mus':torch.zeros(2, 28, device='cuda:0')}
    report = diagnostic.tracking(ZeroPolicy(), path, 'cuda:0', False, epa_recovery=EpaRecovery(tmp_path))
    assert report['count'] == 2 and len(report['windows']) == 2
    failed = next(row for row in report['windows'] if row['simulation_error'])
    assert not failed['success'] and failed['failure_reasons'] == ['EPA_HORIZON']
    assert failed['transitions'] == 0 and failed['coverage'] == 0
    assert failed['mean_reward'] is None
