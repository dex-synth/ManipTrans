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
