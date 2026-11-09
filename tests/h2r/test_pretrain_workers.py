"""Parallel preparation keeps source order, bounded work and strict failures."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from maniptrans_h2r import pretrain_worker
from maniptrans_h2r.motion import WIDTH


def test_ordered_results_bounds_pending_jobs_and_keeps_source_order():
    first_waits = Event()
    submitted = []

    def run(index):
        if index == 0:
            assert first_waits.wait(5)
        else:
            first_waits.set()
        return index

    class Executor:
        def __init__(self, pool):
            self.pool = pool
        def submit(self, function, index):
            submitted.append(index)
            return self.pool.submit(function, index)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = pretrain_worker.ordered_results(Executor(pool), run, range(9), 2)
        assert next(results) == 0
        assert submitted == [0, 1]
        assert list(results) == list(range(1, 9))
    assert submitted == list(range(9))


@pytest.mark.parametrize('allow_incomplete', [False, True])
def test_reference_failure_is_reported_without_silent_fallback(tmp_path, monkeypatch, allow_incomplete):
    arrays = dict(time_s=np.linspace(0, .5, 4), joints_m=np.zeros((4, 21, 3)))
    monkeypatch.setattr(pretrain_worker, 'load_arrays', lambda path: arrays)
    monkeypatch.setattr(pretrain_worker, 'MotionInterpolator', lambda *args:
                        lambda times, **kwargs: np.zeros((len(times), 21, 3)))

    def reference(model, robot, side, human, times, **kwargs):
        if side == 'right':
            raise ValueError('robot_reference_invalid: test workspace failure')
        return np.zeros((len(times), WIDTH), dtype=np.float32), [(0, np.zeros((len(times), 3)))]

    monkeypatch.setattr(pretrain_worker, 'pretraining_reference', reference)
    monkeypatch.setattr(pretrain_worker, 'pack', lambda values, side: values)
    worker = pretrain_worker.ClipWorker.__new__(pretrain_worker.ClipWorker)
    worker.corpus = worker.mano_root = tmp_path
    worker.seed, worker.synthetic_fraction, worker.allow_incomplete = 42, 0., allow_incomplete
    worker.outputs = {side: tmp_path/side for side in ('right', 'left')}
    worker.wrists = {side: SimpleNamespace(model=None, robot={}) for side in worker.outputs}
    for output in worker.outputs.values():
        (output/'.parts').mkdir(parents=True)
    clip = dict(id='clip', path='clip.npz', dataset='taco', source_side='right',
                recording='recording', source_interval=[0, 4])
    result = worker.process((0, clip))
    assert result['right']['samples'] == 0
    assert result['right']['errors'][0]['clip_id'] == 'clip'
    assert 'test workspace failure' in result['right']['errors'][0]['reason']
    assert result['left']['samples'] == (31 if allow_incomplete else 0)
    assert (tmp_path/'right/.parts/0.raw').stat().st_size == 0
