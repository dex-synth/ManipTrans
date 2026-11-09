"""Minimum training clip length is a method decision, not corpus corruption."""

from types import SimpleNamespace
import json

import pytest

from maniptrans_h2r import pretrain
from maniptrans_h2r.io import read_json, seal


def clip(count, duration):
    return dict(id='clip', dataset='taco', recording='recording', source_side='right',
                source_interval=[0, count], num_samples=count, duration_s=duration)


@pytest.mark.parametrize('count,duration,accepted', [
    (1, 0., False), (3, 1., False), (4, .4, False),
    (4, .5, True), (16, .5-5e-11, True), (16, .5-1e-8, False), (30, 1., True),
])
def test_imitator_minimum_length_and_boundary_tolerance(count, duration, accepted):
    item = clip(count, duration)
    selected, excluded = pretrain.select_clips([item])
    if accepted:
        assert selected == [item] and not excluded
    else:
        assert not selected
        assert excluded == [dict(clip_id='clip', **{k: v for k, v in item.items() if k != 'id'}, reason='too_short')]


def test_prepare_records_short_clips_before_mano_interpolation(tmp_path, monkeypatch):
    corpus = tmp_path/'corpus'
    corpus.mkdir()
    (corpus/'clips.jsonl').write_text(json.dumps(clip(1, 0.))+'\n')
    seal(corpus, 'corpus.json', dict(format='h2r-hand-motion-corpus-v1', inventory_complete=True))
    monkeypatch.setattr(pretrain, 'load_task', lambda path: (tmp_path/'task', {}))
    output = tmp_path/'prepared'
    args = SimpleNamespace(corpus=corpus, task=tmp_path/'task', output=output, allow_incomplete=False)
    with pytest.raises(ValueError, match='No clips meet the imitator minimum length'):
        pretrain.prepare(args)
    selection = read_json(output/'clip_selection.json')
    assert selection['selected_clip_count'] == 0
    assert selection['rule'] == pretrain.CLIP_FILTER
    assert selection['excluded_clips'][0]['reason'] == 'too_short'
    assert not (output/'reference_errors.json').exists()
