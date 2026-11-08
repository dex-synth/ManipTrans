"""Balanced window sampling and a bounded 128+2-reference GPU cache."""

import numpy as np
import torch


DATASETS = ("arctic", "oakink2", "hot3d", "taco")
WIDTH = 63+63+13+6


def windows(count, max_samples=2000):
    """Nonoverlapping transitions, shared boundary samples, balanced short tail."""
    lengths = []
    remaining = count-1
    while remaining:
        n = min(remaining, max_samples-1)
        lengths.append(n)
        remaining -= n
    if len(lengths) > 1 and lengths[-1] < 30:
        total = lengths[-2]+lengths[-1]
        lengths[-2:] = [total//2, total-total//2]
    offset = 0
    for length in lengths:
        yield offset, length+1
        offset += length


def pack(reference, side):
    return np.concatenate([reference[f"{side}_human"].reshape(-1, 63),
        reference[f"{side}_human_velocity"].reshape(-1, 63), reference[f"{side}_wrist"], reference["wrist_q"]], axis=1).astype(np.float32)


class MotionCache:
    def __init__(self, root, manifest, count, side, device, seed=42, allow_partial=False):
        self.data = np.load(root / "motion.npy", mmap_mode="r", allow_pickle=False)
        self.windows = manifest["windows"]
        self.count, self.side, self.device = count, side, device
        self.rng = np.random.default_rng(seed)
        self.synthetic_fraction = manifest["synthetic_fraction"]
        self.pools = {}
        self.weights = {}
        for i, w in enumerate(self.windows):
            self.pools.setdefault((w["synthetic"], w["dataset"], w["source_side"]), []).append(i)
        for key, indices in self.pools.items():
            values = np.asarray([self.windows[i]["length"]-1 for i in indices], dtype=float)
            self.weights[key] = values/values.sum()
        expected = {(s, d, h) for s in ([False, True] if self.synthetic_fraction else [False])
                    for d in DATASETS for h in ("left", "right")}
        if not allow_partial and expected - self.pools.keys():
            raise ValueError(f"Missing required sampling pools: {sorted(expected-self.pools.keys())}")
        self.partial = bool(expected - self.pools.keys())
        self.window_ids = np.zeros(count, dtype=np.int64)
        self.positions = np.zeros(count, dtype=np.int64)
        self.lengths = np.zeros(count, dtype=np.int64)
        self.cache_start = np.zeros(count, dtype=np.int64)
        self.cache = torch.empty((count, 130, WIDTH), device=device)
        self.cpu = torch.empty((count, 130, WIDTH), pin_memory=str(device).startswith("cuda"))
        self.visits = np.zeros(len(self.windows), dtype=np.int64)
        self.transitions = np.zeros(len(self.windows), dtype=np.int64)

    def reset(self, ids, *, fixed_windows=None, from_start=False):
        ids = np.asarray(ids, dtype=np.int64)
        for index, env in enumerate(ids):
            if fixed_windows is None:
                synthetic = bool(self.rng.random() < self.synthetic_fraction)
                keys = [k for k in self.pools if k[0] == synthetic]
                if not keys and self.partial:
                    keys = list(self.pools)
                # Uniform dataset and source side; complete pools give exactly 1/4 * 1/2.
                key = keys[int(self.rng.integers(len(keys)))]
                chosen = self.rng.choice(self.pools[key], p=self.weights[key])
            else:
                chosen = fixed_windows[index]
            self.window_ids[env] = chosen
            self.lengths[env] = self.windows[chosen]["length"]
            self.positions[env] = 0 if from_start else min(self.lengths[env]-2, int(.99*self.lengths[env]*self.rng.random()))
            self.visits[chosen] += 1
        self.fill(ids)

    def fill(self, ids):
        for env in ids:
            window = self.windows[self.window_ids[env]]
            self.cache_start[env] = self.positions[env]
            samples = np.minimum(self.positions[env]+np.arange(130), window["length"]-1)+window["offset"]
            self.cpu[env].copy_(torch.from_numpy(np.asarray(self.data[samples]).copy()))
        if len(ids):
            self.cache[torch.as_tensor(ids, device=self.device)] = self.cpu[ids].to(self.device, non_blocking=False)

    def advance(self):
        active = self.positions < self.lengths-1
        np.add.at(self.transitions, self.window_ids[active], 1)
        self.positions[active] += 1

    def sample(self, lookahead=0):
        selected = np.minimum(self.positions+lookahead, self.lengths-1)
        refill = np.flatnonzero(selected-self.cache_start >= 128)
        self.fill(refill)
        indices = torch.as_tensor(selected-self.cache_start, device=self.device)
        row = self.cache[torch.arange(self.count, device=self.device), indices]
        return {f"{self.side}_human": row[:, :63].reshape(-1, 21, 3),
                f"{self.side}_human_velocity": row[:, 63:126].reshape(-1, 21, 3),
                f"{self.side}_wrist": row[:, 126:139], "wrist_q": row[:, 139:145]}

    def state_dict(self):
        return dict(rng=self.rng.bit_generator.state, visits=self.visits, transitions=self.transitions)

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["rng"]
        self.visits[:] = state["visits"]
        self.transitions[:] = state["transitions"]
