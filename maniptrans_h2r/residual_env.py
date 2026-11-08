"""Full-scene residual training with synchronized reference-state initialization."""

import numpy as np
import torch

from .io import load_artifact, load_arrays, load_task, read_json
from .mjlab_env import MjlabVecEnv
from .observations import Reference
from .physics import load_model
from .residual_reward import residual_reward
from .robot_physics import require_clock
from .rsi import sampling_probabilities


class ResidualEnv(MjlabVecEnv):
    def __init__(self, prepared, count, device, seed=42):
        self.path, metadata = load_artifact(prepared)
        if metadata["format"] != "maniptrans-h2r-prepared-v2":
            raise ValueError("Prepare a v2 task before residual training")
        self.task_path, self.task = load_task(self.path / "task")
        require_clock(self.task)
        robot = read_json(self.path / "robot_mapping.json")
        model, bodies, pairs, scene_hash = load_model(self.task_path, self.task, robot, all_contacts=True)
        metadata = dict(metadata, scene_sha256=scene_hash, object_bodies=list(self.task["object"]["parts"].values()))
        self.reference = Reference(load_arrays(self.path / "reference.npz"), device)
        self.rsi = load_arrays(self.path / "rsi.npz")
        self.rsi_probability = sampling_probabilities(len(self.rsi["ticks"]), self.rsi["valid_start"])
        self.rng = np.random.default_rng(seed)
        self.part_indices = [metadata["part_names"].index(p) for p in metadata["scoring_parts"]]
        super().__init__(model, robot, bodies, pairs, count, device, metadata, "residual")
        self.object_qids = [int(model.joint(n).qposadr[0]) for n in self.task["object"]["joints"].values()]
        self.object_vids = [int(model.joint(n).dofadr[0]) for n in self.task["object"]["joints"].values()]
        self.finish_init()

    def target(self, lookahead=0):
        return self.reference.sample((self.tick+lookahead*2).clamp(max=self.task["end_tick"]).float()/120)

    def remaining_ticks(self):
        return self.task["end_tick"]-self.tick

    def reset_states(self, ids):
        indices = self.rng.choice(len(self.rsi_probability), len(ids), p=self.rsi_probability)
        for key in ("qpos", "qvel"):
            self.views[key][ids] = torch.tensor(self.rsi[key][indices], dtype=torch.float32, device=self.device)
        self.tick[ids] = torch.tensor(self.rsi["ticks"][indices], dtype=torch.long, device=self.device)
        self.views["time"][ids] = self.tick[ids].float()/120

    def reward(self, state, target, power, actions):
        return residual_reward(self.robot, self.part_indices, state, target, power, actions,
                               self.elapsed_ticks, self.scale, self.contact_history)

    def sampler_state(self):
        return self.rng.bit_generator.state

    def load_sampler_state(self, state):
        self.rng.bit_generator.state = state
