"""EPA recovery must isolate physics, returns, normalization and resume identity."""

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lib.rl.agent import PPOAgent
from lib.rl.moving_avg import RunningMeanStd
from lib.rl.numerics import normalize_valid
from maniptrans_h2r.buffers import OverflowType, SimulationOverflow, check_overflow
from maniptrans_h2r.io import digest
from maniptrans_h2r.recovery import EPA_RECOVERY, EpaRecovery, STATE_KEYS
from maniptrans_h2r.training import TRAINING_REVISION, ReproductionAgent, recipe_identity, recovery_resume_changes


def binding(recipe):
    return dict(prepared_id='same-corpus', bases={}, recipe_id=digest(recipe_identity(recipe)))


@pytest.fixture
def recipes():
    old = dict(worlds=8192, seed=42, training_revision=TRAINING_REVISION, control={'schema':'v3'},
               learning_rate=.0005, contact_capacity=256, constraint_capacity=2048)
    return old, dict(old, epa_recovery=deepcopy(EPA_RECOVERY))


def test_epa_resume_only_adds_authenticated_policy(recipes):
    old, new = recipes
    assert recovery_resume_changes(binding(old), binding(new), old, new) == {}
    with pytest.raises(ValueError):
        recovery_resume_changes(binding(old), binding(new), dict(old, seed=0), new)
    with pytest.raises(ValueError):
        recovery_resume_changes(binding(old), dict(binding(new), prepared_id='other'), old, new)
    with pytest.raises(ValueError):
        recovery_resume_changes(binding(new), binding(new), new, new)


@pytest.mark.parametrize('change', [dict(worlds=4096), dict(seed=0), dict(training_revision=TRAINING_REVISION+1),
    dict(learning_rate=.001), dict(extra=True), dict(control={'schema':'other'}),
    dict(contact_capacity=128), dict(epa_recovery=dict(EPA_RECOVERY, max_skipped_episodes=999))])
def test_epa_resume_rejects_other_recipe_changes(recipes, change):
    old, new = recipes
    new.update(change)
    with pytest.raises(ValueError):
        recovery_resume_changes(binding(old), binding(new), old, new)


def test_epa_resume_does_not_implicitly_authorize_capacity_changes(recipes):
    old, new = recipes
    new['contact_capacity'] = 512
    with pytest.raises(ValueError, match='also require'):
        recovery_resume_changes(binding(old), binding(new), old, new)
    assert recovery_resume_changes(binding(old), binding(new), old, new, True) == {
        'contact_capacity': {'before':256, 'after':512}}


def test_epa_resume_requires_opt_in_before_loading_optimizer(tmp_path, recipes):
    old, new = recipes
    path = tmp_path/'resume.pth'
    torch.save({'h2r': {'binding': binding(old), 'recipe': old}}, path)
    agent = ReproductionAgent.__new__(ReproductionAgent)
    agent.ppo_device, agent.binding, agent.recipe = 'cpu', binding(new), new
    with pytest.raises(ValueError, match='binding differs'):
        agent.restore_local(path)


def test_epa_resume_does_not_silently_reset_missing_counters(tmp_path, recipes):
    _, recipe = recipes
    path = tmp_path/'resume.pth'
    torch.save({'h2r': {'binding': binding(recipe), 'recipe': recipe}}, path)
    agent = ReproductionAgent.__new__(ReproductionAgent)
    agent.ppo_device, agent.binding, agent.recipe = 'cpu', binding(recipe), recipe
    with pytest.raises(ValueError, match='missing recovery counters'):
        agent.restore_local(path, recover_epa=True)


def test_gae_truncates_at_failure_and_bootstraps_last_valid_state():
    agent = ReproductionAgent.__new__(ReproductionAgent)
    agent.recipe, agent.horizon_length, agent.gamma, agent.tau = {'epa_recovery':EPA_RECOVERY}, 4, .9, .95
    masks = torch.ones(4, 2)
    masks[1, 0] = 0
    agent.experience_buffer = SimpleNamespace(tensor_dict={'error_masks': masks})
    values = torch.tensor([[2., 2.], [3., 3.], [20., 4.], [21., 5.]])[:, :, None]
    rewards = torch.ones_like(values)
    dones = torch.zeros(4, 2)
    dones[2, 0] = 1  # Row 2 is a fresh episode after the invalid row 1.
    result = agent.discount_values(torch.zeros(2), torch.tensor([[22.], [6.]]), dones, values, rewards)
    assert result[0, 0, 0] == pytest.approx(1+.9*3-2)
    assert result[1, 0, 0] == 0
    baseline = PPOAgent.discount_values(agent, torch.zeros(2), torch.tensor([[22.], [6.]]), dones, values, rewards)
    torch.testing.assert_close(result[:, 1], baseline[:, 1])
    rewards[1, 0] = 1e8
    values[2:, 0] *= 100
    changed = agent.discount_values(torch.zeros(2), torch.tensor([[2200.], [6.]]), dones, values, rewards)
    torch.testing.assert_close(result[:2], changed[:2])


@pytest.mark.parametrize('bad_step', [0, 3])
def test_gae_handles_rollout_boundaries(bad_step):
    agent = ReproductionAgent.__new__(ReproductionAgent)
    agent.recipe, agent.horizon_length, agent.gamma, agent.tau = {'epa_recovery':EPA_RECOVERY}, 4, .9, .95
    masks = torch.ones(4, 1); masks[bad_step] = 0
    agent.experience_buffer = SimpleNamespace(tensor_dict={'error_masks':masks})
    values = torch.ones(4, 1, 1)*2
    rewards = torch.ones_like(values)
    dones = torch.zeros(4, 1)
    if bad_step < 3:
        dones[bad_step+1] = 1
    result = agent.discount_values(torch.ones(1), torch.ones(1, 1)*100, dones, values, rewards)
    assert result[bad_step] == 0
    if bad_step:
        assert result[bad_step-1] == pytest.approx(1+.9*2-2)


def test_invalid_rows_do_not_update_value_moments():
    normalizer = RunningMeanStd((1,))
    expected = deepcopy(normalizer)
    values = torch.tensor([[1.], [1e20], [3.]])
    valid = torch.tensor([True, False, True])
    normalized = normalize_valid(normalizer, values, valid)
    expected(values[valid]); expected.eval()
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(value, normalizer.state_dict()[key])
    torch.testing.assert_close(normalized[valid], expected(values[valid]))


def test_invalid_minibatch_rows_never_reach_policy_or_input_normalizer(monkeypatch):
    agent = ReproductionAgent.__new__(ReproductionAgent)
    agent.recipe = {'epa_recovery':EPA_RECOVERY}
    agent.model = SimpleNamespace(running_mean_std=RunningMeanStd((1,)))
    batch = dict(error_masks=torch.tensor([1., 0., 1.]), mu=torch.zeros(3, 2), sigma=torch.ones(3, 2),
                 obs={'a':torch.tensor([[2.], [1e20], [4.]])}, actions=torch.zeros(3, 2))
    def update(self, rows):
        assert len(rows['actions']) == 2
        assert rows['obs']['a'].tolist() == [[2.], [4.]]
        self.model.running_mean_std(rows['obs']['a'])
        return (0, 0, 0, 0, .001, 1, rows['mu']+3, rows['sigma']+2, 0)
    monkeypatch.setattr(PPOAgent, 'train_actor_critic', update)
    result = agent.train_actor_critic(batch)
    assert result[6].tolist() == [[3., 3.], [0., 0.], [3., 3.]]
    assert result[7].tolist() == [[3., 3.], [1., 1.], [3., 3.]]
    assert agent.model.running_mean_std.count == 3  # Initial pseudo-count + two valid rows.


def test_fully_invalid_minibatch_does_not_step_optimizer(monkeypatch):
    agent = ReproductionAgent.__new__(ReproductionAgent)
    agent.recipe, agent.ppo_device, agent.last_lr = {'epa_recovery':EPA_RECOVERY}, 'cpu', .001
    monkeypatch.setattr(PPOAgent, 'train_actor_critic', lambda *args: pytest.fail('must not update'))
    result = agent.train_actor_critic(dict(error_masks=torch.zeros(2), mu=torch.zeros(2, 3), sigma=torch.ones(2, 3)))
    assert result[0] == 0 and result[4] == .001


def make_env(count=4):
    views = {key: torch.ones(count, 2) for key in STATE_KEYS}
    return SimpleNamespace(count=count, device='cpu', common_steps=20, views=views,
        motion=SimpleNamespace(window_ids=np.zeros(count, dtype=int), positions=np.ones(count, dtype=int),
                               windows=[dict(clip_id='reference', recording='recording')]),
        data=SimpleNamespace(overflow=torch.zeros(count, dtype=torch.int)))


def overflow(ids=(1,), flags=None, phase='step'):
    return SimulationOverflow(dict(phase=phase, flags=flags or {'EPA_HORIZON':len(ids)},
        world_count=len(ids), world_ids=list(ids[:16]), limits={}), torch.tensor(ids))


def test_recovery_records_snapshot_counts_once_and_preserves_resume_budget(tmp_path, monkeypatch):
    monkeypatch.setattr('maniptrans_h2r.recovery.wp.to_torch', lambda value:value)
    env = make_env()
    recovery = EpaRecovery(tmp_path/'first')
    invalid = torch.zeros(4, dtype=torch.bool)
    before = {key:value.clone() for key, value in env.views.items()}
    env.data.overflow[1] = int(OverflowType.EPA_HORIZON)
    ids = recovery.handle(env, overflow(), before, invalid, 0)
    invalid[ids] = True
    recovery.handle(env, overflow(), before, invalid, 1)
    assert recovery.skipped_episodes == 1 and recovery.events == 2
    assert env.data.overflow.sum() == 0
    snapshot = torch.load(tmp_path/'first/event-000001.pth', weights_only=True)
    torch.testing.assert_close(snapshot['qpos'], before['qpos'][[1]])
    row = json.loads((tmp_path/'first/events.jsonl').read_text().splitlines()[0])
    assert row['references'][0]['reference']['clip_id'] == 'reference'
    restored = EpaRecovery(tmp_path/'second')
    restored.load_state_dict(deepcopy(recovery.state_dict()))
    assert restored.state_dict() == recovery.state_dict()
    restored.recent.append((env.common_steps, 31))
    with pytest.raises(SimulationOverflow) as caught:
        restored.handle(env, overflow(), before, torch.zeros_like(invalid), 0)
    assert caught.value.report['recovery_refused'] == 'too_many_skipped_episodes'
    env.common_steps += EPA_RECOVERY['window_controls']
    restored.handle(env, overflow(), before, torch.zeros_like(invalid), 0)


@pytest.mark.parametrize('flags,phase', [({'NEFC':1}, 'step'), ({'EPA_HORIZON':1,'CCD':1}, 'step'),
    ({'UNKNOWN':1}, 'step'), ({'EPA_HORIZON':1}, 'reset')])
def test_only_local_step_epa_is_recoverable(tmp_path, flags, phase):
    env = make_env()
    with pytest.raises(SimulationOverflow):
        EpaRecovery(tmp_path).handle(env, overflow(flags=flags, phase=phase), env.views, torch.zeros(4, dtype=torch.bool), 0)


def test_multiple_worlds_and_nonfinite_state_still_stop(tmp_path):
    env = make_env()
    recovery = EpaRecovery(tmp_path)
    with pytest.raises(SimulationOverflow) as caught:
        recovery.handle(env, overflow((0, 1)), env.views, torch.zeros(4, dtype=torch.bool), 0)
    assert caught.value.report['recovery_refused'] == 'too_many_affected_worlds'
    env.views['qpos'][1, 0] = float('nan')
    with pytest.raises(FloatingPointError):
        recovery.handle(env, overflow(), env.views, torch.zeros(4, dtype=torch.bool), 0)


def test_recovery_uses_full_world_ids_not_truncated_report(monkeypatch):
    monkeypatch.setattr('maniptrans_h2r.buffers.wp.to_torch', lambda value:value)
    data = SimpleNamespace(overflow=torch.full((20,), int(OverflowType.EPA_HORIZON)),
        nefc=torch.zeros(20), nacon=torch.zeros(1), naconmax=100, njmax=100, naccdmax=100, njmax_nnz=100)
    with pytest.raises(SimulationOverflow) as caught:
        check_overflow(data, 'step')
    assert len(caught.value.report['world_ids']) == 16
    assert caught.value.world_ids.tolist() == list(range(20))
