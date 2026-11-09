"""Allocation changes must not relax task or learning-recipe identity checks."""

from types import SimpleNamespace

import pytest
import torch

from maniptrans_h2r import buffers
from maniptrans_h2r.io import digest
from maniptrans_h2r.training import capacity_resume_changes, recipe_identity, ReproductionAgent


@pytest.fixture
def recipes():
    old = dict(contact_capacity=128, constraint_capacity=1024, worlds=8192,
               learning_rate=.0005, training_revision=7, seed=42,
               control={'schema': 'virtual-wrist-position-v3'}, epochs=1000,
               save_every=1000, diagnostic_every=250)
    new = dict(old, constraint_capacity=2048, epochs=2000, save_every=100)
    return old, new


def binding(recipe, **overrides):
    return dict(prepared_id='sealed-reference', recipe_id=digest(recipe_identity(recipe)),
                bases={}, **overrides)


def test_capacity_resume_allows_only_allocations_and_existing_budget_exceptions(recipes):
    old, new = recipes
    assert capacity_resume_changes(binding(old), binding(new), old, new) == {
        'constraint_capacity': {'before': 1024, 'after': 2048}}
    assert capacity_resume_changes(binding(old), binding(old), old, old) == {}


@pytest.mark.parametrize('change', [dict(worlds=4096), dict(learning_rate=.001), dict(seed=43),
    dict(training_revision=8), dict(control={'schema':'changed'}), dict(contact_capacity=64),
    dict(constraint_capacity=512), dict(extra_field=True)])
def test_capacity_resume_rejects_learning_changes_reduced_limits_and_new_fields(recipes, change):
    old, new = recipes
    new.update(change)
    with pytest.raises(ValueError):
        capacity_resume_changes(binding(old), binding(new), old, new)


@pytest.mark.parametrize('key,value', [('prepared_id','other-reference'), ('bases', {'right':'other-base'})])
def test_capacity_resume_rejects_changed_inputs(recipes, key, value):
    old, new = recipes
    target = binding(new)
    target[key] = value
    with pytest.raises(ValueError, match='binding differs'):
        capacity_resume_changes(binding(old), target, old, new)


def test_capacity_resume_requires_original_recipe_to_match_checkpoint_hash(recipes):
    old, new = recipes
    changed = dict(old, worlds=4096)
    for invalid in (None, changed):
        with pytest.raises(ValueError, match='hash-matching'):
            capacity_resume_changes(binding(old), binding(new), invalid, new)
    with pytest.raises(ValueError, match='Current resume recipe hash'):
        capacity_resume_changes(binding(old), binding(old), old, new)


@pytest.mark.parametrize('bad', [0, -1, True, 1024.5, '1024'])
def test_capacity_values_are_positive_integers(recipes, bad):
    old, new = recipes
    new['constraint_capacity'] = bad
    with pytest.raises(ValueError, match='positive integers'):
        capacity_resume_changes(binding(old), binding(new), old, new)


def test_restoring_increased_capacity_requires_explicit_opt_in(tmp_path, recipes):
    old, new = recipes
    path = tmp_path/'resume.pth'
    torch.save({'h2r': {'binding': binding(old), 'recipe': old}}, path)
    agent = ReproductionAgent.__new__(ReproductionAgent)
    agent.ppo_device, agent.binding, agent.recipe = 'cpu', binding(new), new
    with pytest.raises(ValueError, match='binding differs'):
        agent.restore_local(path)


def test_restore_rejects_tampered_legacy_sidecar_before_loading_state(tmp_path, recipes):
    import yaml
    old, new = recipes
    checkpoint, config = tmp_path/'resume.pth', tmp_path/'resolved_config.yaml'
    torch.save({'h2r': {'binding': binding(old)}}, checkpoint)
    config.write_text(yaml.safe_dump(dict(old, seed=99)))
    agent = ReproductionAgent.__new__(ReproductionAgent)
    agent.ppo_device, agent.binding, agent.recipe = 'cpu', binding(new), new
    with pytest.raises(ValueError, match='does not match checkpoint recipe hash'):
        agent.restore_local(checkpoint, allow_capacity_increase=True, resume_config=config)


def test_overflow_reports_all_flags_counts_and_bounded_world_examples(monkeypatch):
    monkeypatch.setattr(buffers.wp, 'to_torch', lambda x: x)
    masks = torch.full((20,), int(buffers.OverflowType.NEFC | buffers.OverflowType.CONTACT_MATCH))
    masks[3] |= 1 << 20
    data = SimpleNamespace(overflow=masks, nefc=torch.full((20,), 1100), nacon=torch.tensor([2500]),
                           naconmax=2560, njmax=1024, naccdmax=5120, njmax_nnz=4096)
    with pytest.raises(buffers.SimulationOverflow, match='NEFC') as caught:
        buffers.check_overflow(data, 'reset')
    report = caught.value.report
    assert report['flags'] == {'NEFC':20, 'CONTACT_MATCH':20, 'UNKNOWN':1}
    assert report['phase'] == 'reset' and report['world_count'] == 20
    assert report['world_ids'] == list(range(16))
    assert report['nefc'] == [1100]*16 and report['contacts_total'] == 2500
    assert report['masks'][3] == int(masks[3])


def test_no_overflow_does_not_read_diagnostic_arrays(monkeypatch):
    monkeypatch.setattr(buffers.wp, 'to_torch', lambda x: x)
    buffers.check_overflow(SimpleNamespace(overflow=torch.zeros(4)), 'step')


def test_cli_capacities_keep_stage_defaults_and_require_resume_for_resume_flags():
    import argparse
    from maniptrans_h2r.training import training_arguments, training_capacities
    parser = argparse.ArgumentParser()
    training_arguments(parser, 'imitator')
    args = parser.parse_args(['--prepared','input','--output','output'])
    assert training_capacities(args) == dict(contact_capacity=128, constraint_capacity=1024)
    args = parser.parse_args(['--prepared','input','--output','output','--contact-capacity','256',
                              '--constraint-capacity','2048','--allow-capacity-increase'])
    with pytest.raises(ValueError, match='require --resume'):
        training_capacities(args)
    args.resume = 'checkpoint.pth'
    assert training_capacities(args) == dict(contact_capacity=256, constraint_capacity=2048)
