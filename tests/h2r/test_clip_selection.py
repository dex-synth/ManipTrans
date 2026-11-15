"""Minimum training clip length is a method decision, not corpus corruption."""

from types import SimpleNamespace
import json

import pytest

from maniptrans_h2r import pretrain
from maniptrans_h2r.h2r_io import mano
from maniptrans_h2r.io import digest, read_json, sha256, write_json


def clip(count, duration):
    return dict(id='clip', dataset='taco', recording='recording', source_side='right',
                source_interval=[0, count], num_samples=count, duration_s=duration)


def write_corpus(corpus, clips):
    corpus.mkdir()
    (corpus/'clips.jsonl').write_text(''.join(json.dumps(item)+'\n' for item in clips))
    meta = dict(format='h2r-hand-motion-corpus-v1', inventory_complete=True,
                files={'clips.jsonl': sha256(corpus/'clips.jsonl')})
    write_json(corpus/'corpus.json', dict(meta, id=digest(meta)))
    return corpus


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
    corpus = write_corpus(tmp_path/'corpus', [dict(clip(1, 0.), tip_vertex_ids=list(mano.TIP_VERTEX_IDS))])
    monkeypatch.setattr(pretrain, 'load_task', lambda path: SimpleNamespace(path=tmp_path/'task', hand_sides=('right',)))
    output = tmp_path/'prepared'
    args = SimpleNamespace(corpus=corpus, task=tmp_path/'task', output=output, allow_incomplete=False, side='right')
    with pytest.raises(ValueError, match='No clips meet the imitator minimum length'):
        pretrain.prepare(args)
    selection = read_json(output/'clip_selection.json')
    assert selection['selected_clip_count'] == 0
    assert selection['rule'] == pretrain.CLIP_FILTER
    assert selection['excluded_clips'][0]['reason'] == 'too_short'
    assert not (output/'reference_errors.json').exists()


def test_corpus_with_per_dataset_fingertips_is_rejected(tmp_path):
    # Corpora exported before the common MANO convention used TACO's own fingertip vertices.
    corpus = write_corpus(tmp_path/'corpus', [dict(clip(30, 1.), tip_vertex_ids=[745, 317, 445, 556, 673])])
    args = SimpleNamespace(corpus=corpus, task=tmp_path/'task', output=tmp_path/'prepared', allow_incomplete=False, side='right')
    with pytest.raises(ValueError, match='predates the common MANO convention'):
        pretrain.prepare(args)
    assert not args.output.exists()


def test_side_must_be_a_hand_of_the_task(tmp_path, monkeypatch):
    corpus = write_corpus(tmp_path/'corpus', [dict(clip(30, 1.), tip_vertex_ids=list(mano.TIP_VERTEX_IDS))])
    monkeypatch.setattr(pretrain, 'load_task', lambda path: SimpleNamespace(path=tmp_path/'task', hand_sides=('right',)))
    args = SimpleNamespace(corpus=corpus, task=tmp_path/'task', output=tmp_path/'prepared', allow_incomplete=False, side='both')
    with pytest.raises(ValueError, match="this one has \\['right'\\]"):
        pretrain.prepare(args)
