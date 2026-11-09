"""Clip-level CPU preparation, with bounded scheduling and ordered results."""

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager, ExitStack
import math
import multiprocessing
import os
from pathlib import Path

import numpy as np

from .hand_reference import MotionInterpolator, WristReference, pretraining_reference
from .io import digest, load_arrays, read_json
from .motion import pack


class ClipWorker:
    def __init__(self, corpus, outputs, mano_root, seed, synthetic_fraction, allow_incomplete):
        import mujoco
        import torch
        torch.set_num_threads(1)
        self.corpus, self.outputs, self.mano_root = Path(corpus), outputs, mano_root
        self.seed, self.synthetic_fraction, self.allow_incomplete = seed, synthetic_fraction, allow_incomplete
        self.wrists = {}
        for side, output in outputs.items():
            model = mujoco.MjModel.from_xml_path(str(Path(output) / "robot_scene.xml"))
            robot = read_json(Path(output) / "robot_mapping.json")
            self.wrists[side] = WristReference(model, robot, side)

    def process(self, job):
        index, clip = job
        arrays = load_arrays(self.corpus / clip["path"])
        motion = MotionInterpolator(arrays, clip, self.mano_root)
        if np.max(np.abs(motion(arrays["time_s"])-arrays["joints_m"])) > 1e-5:
            raise ValueError(f"MANO reconstruction differs: {clip['id']}")
        duration = float(arrays["time_s"][-1])
        rng = np.random.default_rng(int(digest([self.seed, clip["id"]])[:16], 16))
        recipes = [(False, None, duration)]
        if self.synthetic_fraction:
            for _ in range(max(1, math.ceil(duration/10))):
                start = rng.uniform(0, max(0, duration-.5))
                length = rng.uniform(.5, min(5, duration-start))
                recipes.append((True, [start, start+length], length))
        result = {side: dict(records=[], errors=[], samples=0,
                            path=str(Path(output) / ".parts" / f"{index}.raw"))
                  for side, output in self.outputs.items()}
        with ExitStack() as stack:
            streams = {side: stack.enter_context(open(row["path"], "wb")) for side, row in result.items()}
            for synthetic, endpoints, length in recipes:
                times = np.arange(int(np.floor(length*60+1e-9))+1)/60
                try:
                    points = motion(times, endpoints=endpoints)
                except ValueError as exc:
                    for row in result.values():
                        row["errors"].append(self.error(clip, synthetic, endpoints, exc))
                    if not self.allow_incomplete:
                        break
                    continue
                for side, wrist in self.wrists.items():
                    row = result[side]
                    human = points if clip["source_side"] == side else points*np.array([-1, 1, 1], dtype=np.float32)
                    try:
                        reference, segments = pretraining_reference(wrist.model, wrist.robot, side, human, times, wrist=wrist)
                        values = pack(reference, side)
                    except ValueError as exc:
                        row["errors"].append(self.error(clip, synthetic, endpoints, exc))
                        if not self.allow_incomplete:
                            return result
                        continue
                    for start, angles in segments:
                        count = len(angles)
                        chunk = values[start:start+count].copy()
                        chunk[:, -3:] = angles
                        streams[side].write(chunk.tobytes())
                        row["records"].append(dict(offset=row["samples"], length=count, source_sample_start=start,
                            dataset=clip["dataset"], source_side=clip["source_side"], synthetic=synthetic,
                            clip_id=clip["id"], recording=clip["recording"], endpoints=endpoints, unused_tail_s=length-times[-1]))
                        row["samples"] += count
        return result

    @staticmethod
    def error(clip, synthetic, endpoints, exc):
        return dict(clip_id=clip["id"], dataset=clip["dataset"], source_interval=clip["source_interval"],
                    synthetic=synthetic, endpoints=endpoints, reason=str(exc))


_worker = None


def initialize(*args):
    global _worker
    _worker = ClipWorker(*args)


def process(job):
    return _worker.process(job)


def ordered_results(executor, function, jobs, capacity):
    """Bound queued clips and temporary shards while preserving source order."""
    jobs = iter(jobs)
    pending = deque()
    for _ in range(capacity):
        job = next(jobs, None)
        if job is None:
            break
        pending.append(executor.submit(function, job))
    while pending:
        yield pending.popleft().result()
        job = next(jobs, None)
        if job is not None:
            pending.append(executor.submit(function, job))


@contextmanager
def worker_pool(workers, initargs):
    # Set before spawn imports NumPy/SciPy/Torch. Keep library pools from
    # multiplying the clip-level process count; restore the caller's settings.
    limits = {name: "1" for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
    previous = {name: os.environ.get(name) for name in limits}
    os.environ.update(limits)
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=initialize, initargs=initargs) as executor:
            yield executor
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
