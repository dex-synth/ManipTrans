"""Regression coverage for bounded wrist RSI and actual training completion."""

from copy import deepcopy
import json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from maniptrans_h2r.hand_reference import WristLimitError, WristReference
from maniptrans_h2r.imitator_env import ImitatorEnv, WRIST_RSI, sample_wrist_pose
from maniptrans_h2r.motion import MotionCache
from maniptrans_h2r.training import ReproductionAgent


@pytest.fixture
def wrist_pose():
    wrist = WristReference.__new__(WristReference)
    wrist.origin = np.zeros(3)
    wrist.translation_axes = wrist.parent_rotation = wrist.local_rotation = np.eye(3)
    wrist.lower = np.array([-5.]*3 + [-6.2]*3)
    wrist.upper = -wrist.lower
    anchor = np.array([.1, .2, 6.19])
    reference = np.r_[np.zeros(3), Rotation.from_euler('XYZ', anchor).as_quat(scalar_first=True), np.zeros(6)].astype(np.float32)
    return wrist, reference, anchor


class PositionNoise:
    """Force a known number of out-of-workspace candidates."""
    def __init__(self, rejected):
        self.rejected, self.calls = rejected, 0

    def normal(self, loc, scale, size=None):
        if size == 3:
            self.calls += 1
            return np.full(3, 10. if self.calls <= self.rejected else 0.)
        return 0.

    def random(self, size):
        return np.ones(size)


def test_rsi_retries_beyond_eight_and_keeps_the_reference_branch(wrist_pose):
    wrist, reference, anchor = wrist_pose
    rng = PositionNoise(8)
    q, rejected, fallback = sample_wrist_pose(wrist, reference, anchor, rng)
    assert rng.calls == 9 and rejected == 8 and not fallback
    np.testing.assert_allclose(q[3:], anchor, atol=1e-6)


def test_rsi_exhaustion_returns_the_validated_reference(wrist_pose):
    wrist, reference, anchor = wrist_pose
    rng = PositionNoise(1000)
    q, rejected, fallback = sample_wrist_pose(wrist, reference, anchor, rng)
    assert fallback and rng.calls == rejected == WRIST_RSI['max_attempts']
    assert np.all(q >= wrist.lower) and np.all(q <= wrist.upper)
    np.testing.assert_allclose(q[:3], reference[:3], atol=0)
    np.testing.assert_allclose(q[3:], anchor, atol=1e-6)


def test_rsi_invalid_reference_is_rejected_before_noise(wrist_pose):
    wrist, reference, anchor = wrist_pose
    reference[0] = 6
    rng = PositionNoise(0)
    with pytest.raises(WristLimitError, match='workspace'):
        sample_wrist_pose(wrist, reference, anchor, rng)
    assert rng.calls == 0


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_rsi_nonfinite_reference_is_not_a_retryable_limit_error(wrist_pose, bad):
    wrist, reference, anchor = wrist_pose
    reference[0] = bad
    with pytest.raises(ValueError, match='Non-finite') as caught:
        sample_wrist_pose(wrist, reference, anchor, PositionNoise(0))
    assert not isinstance(caught.value, WristLimitError)


def test_rsi_solver_errors_are_not_swallowed(wrist_pose, monkeypatch):
    wrist, reference, anchor = wrist_pose
    solve = wrist.solve
    calls = []
    def broken(*args, **kwargs):
        calls.append(1)
        if len(calls) > 1:
            raise ValueError('Wrist Euler inverse failed FK check')
        return solve(*args, **kwargs)
    monkeypatch.setattr(wrist, 'solve', broken)
    with pytest.raises(ValueError, match='FK check'):
        sample_wrist_pose(wrist, reference, anchor, PositionNoise(0))
    assert len(calls) == 2


def test_rsi_normal_path_preserves_noise_and_rng(wrist_pose):
    wrist, reference, anchor = wrist_pose
    # Away from the limit, compare the previous sampling order exactly.
    anchor[:] = [.1, .2, .3]
    reference[3:7] = Rotation.from_euler('XYZ', anchor).as_quat(scalar_first=True)
    rng, expected_rng = np.random.default_rng(43), np.random.default_rng(43)
    p = reference[:3].copy()
    p += expected_rng.normal(0, .01, 3)
    axis = expected_rng.random(3)
    r = Rotation.from_rotvec(axis/np.linalg.norm(axis)*expected_rng.normal(0, np.pi/18))
    r = r*Rotation.from_quat(reference[3:7], scalar_first=True)
    expected = wrist.solve(p[None], r.as_matrix()[None], previous=anchor)[0]
    q, rejected, fallback = sample_wrist_pose(wrist, reference, anchor, rng)
    np.testing.assert_array_equal(q, expected)
    assert rejected == 0 and not fallback
    assert rng.bit_generator.state == expected_rng.bit_generator.state


def test_rsi_counters_and_rng_survive_sampler_restore(tmp_path):
    np.save(tmp_path/'motion.npy', np.zeros((31, 145), dtype=np.float32))
    manifest = dict(synthetic_fraction=0., windows=[dict(synthetic=False, dataset='arctic', source_side='right', length=31, offset=0)])
    env = ImitatorEnv.__new__(ImitatorEnv)
    env.motion = MotionCache(tmp_path, manifest, 1, 'right', 'cpu', allow_partial=True)
    env.rng = np.random.default_rng(43)
    env.wrist_rsi_stats = dict(resets=12, rejected_noise_samples=9, fallbacks=2)
    state = deepcopy(env.sampler_state())
    expected = env.rng.random(5)
    env.wrist_rsi_stats['fallbacks'] += 1
    env.load_sampler_state(state)
    np.testing.assert_array_equal(env.rng.random(5), expected)
    assert env.wrist_rsi_stats == state['wrist_rsi']


@pytest.mark.parametrize('epoch,period', [(37, 250), (10000, 250), (37, 0)])
@pytest.mark.parametrize('already_diagnosed', [False, True])
def test_final_diagnostics_run_once_at_actual_stop(tmp_path, monkeypatch, epoch, period, already_diagnosed):
    from maniptrans_h2r import diagnostic
    calls = []
    def tracking(model, *args, stochastic=False, **kwargs):
        model.eval()
        calls.append(stochastic)
        torch.rand(3)
        return dict(stochastic=stochastic)
    monkeypatch.setattr(diagnostic, 'tracking', tracking)
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [])
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', lambda values: None)
    agent = ReproductionAgent.__new__(ReproductionAgent)
    agent.model = torch.nn.Linear(1, 1).train()
    agent.epoch_num, agent.last_diagnostic, agent.last_final_diagnostic = epoch, -1, -1
    agent.recipe = dict(save_every=1000, diagnostic_every=period)
    agent.diagnostic_callback = diagnostic.diagnostic_callback(tmp_path, tmp_path, 'cpu', False)
    rng = torch.get_rng_state().clone()
    before = deepcopy(agent.model.state_dict())
    if already_diagnosed:
        agent.diagnostic()
    agent.diagnostic(final=True)
    agent.diagnostic(final=True)
    assert calls == [False, True]
    assert json.loads((tmp_path/'diagnostics/final-rsi-sampled.json').read_text())['stochastic']
    assert (tmp_path/f'diagnostics/epoch-{epoch}.json').exists()
    assert agent.last_final_diagnostic == epoch and agent.model.training
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in before.items():
        torch.testing.assert_close(value, agent.model.state_dict()[key], rtol=0, atol=0)


def test_old_resume_binding_is_rejected_before_loading_state(tmp_path):
    agent = ReproductionAgent.__new__(ReproductionAgent)
    agent.ppo_device = 'cpu'
    agent.binding = dict(recipe_id='current-rsi-recipe')
    path = tmp_path/'old-resume.pth'
    torch.save(dict(h2r=dict(binding=dict(recipe_id='old-rsi-recipe'))), path)
    with pytest.raises(ValueError, match='Earlier checkpoints cannot be continued'):
        agent.restore_local(path)


def test_residual_final_callback_does_not_repeat_same_epoch_tracking(tmp_path, monkeypatch):
    from maniptrans_h2r import diagnostic
    calls = []
    def tracking(*args, **kwargs):
        calls.append(1)
        return dict(status='completed')
    monkeypatch.setattr(diagnostic, 'residual_tracking', tracking)
    callback = diagnostic.residual_diagnostic_callback(tmp_path, tmp_path, 'cpu')
    callback(None, 37, mean=True, final=False)
    callback(None, 37, mean=False, final=True)
    assert calls == [1]
    assert (tmp_path/'diagnostics/epoch-37.json').exists()
