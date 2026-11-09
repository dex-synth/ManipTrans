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
