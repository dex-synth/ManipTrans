"""Full-scene residual training with synchronized reference-state initialization."""

import numpy as np
import torch

from .clock import DECIMATION, PHYSICS_DT_S, training_ticks
from .h2r_io import load_task
from .io import load_artifact, load_arrays, read_json
from .mjlab_env import MjlabVecEnv
from .observations import Reference
from .physics import load_model
from .residual_reward import residual_reward
from .rsi import sampling_probabilities
from .robot import initial_qpos


class ResidualEnv(MjlabVecEnv):
    def __init__(self, prepared, count, device, seed=42, *, from_start=False, curriculum=True, buffer_capacities=None):
        self.path, metadata = load_artifact(prepared, "prepared")
        self.task = load_task(self.path / "task")
        if self.task.task_id != metadata["task_id"]:
            raise ValueError("Prepared reference belongs to a different task")
        binding = self.task.object_binding
        robot = read_json(self.path / "robot_mapping.json")
        model, bodies, pairs, scene_hash = load_model(self.task, robot)
        metadata = dict(metadata, scene_sha256=scene_hash)
        self.end_tick = training_ticks(self.task.duration_s)
        self.reference = Reference(load_arrays(self.path / "reference.npz"), device)
        self.rsi = load_arrays(self.path / "rsi.npz")
        self.rsi_ticks = np.rint(self.rsi["time_s"]/PHYSICS_DT_S).astype(np.int64)
        self.rsi_probability = sampling_probabilities(len(self.rsi_ticks), self.rsi["valid_start"])
        self.rng = np.random.default_rng(seed)
        self.from_start = from_start
        self.initialization = read_json(self.path / "initialization.json")
        if from_start and not self.initialization["valid"]:
            raise ValueError("Initial robot pose failed collision admission")
        self.part_indices = [metadata["part_names"].index(p) for p in metadata["scoring_parts"]]
        super().__init__(model, robot, bodies, pairs, count, device, metadata, "residual", buffer_capacities)
        self.object_qids = [int(model.joint(n).qposadr[0]) for n in binding["joints"].values()]
        self.object_vids = [int(model.joint(n).dofadr[0]) for n in binding["joints"].values()]
        if curriculum:
            from .curriculum import PhysicsCurriculum
            self.curriculum = PhysicsCurriculum(self)
        self.finish_init()

    def target(self, lookahead=0):
        return self.reference.sample((self.tick+lookahead*DECIMATION).clamp(max=self.end_tick).float()*PHYSICS_DT_S)

    def remaining_ticks(self):
        return self.end_tick-self.tick

    def reset_states(self, ids):
        if self.from_start:
            qpos = initial_qpos(self.cpu, self.task)
            qpos[self.robot["qpos_indices"]] = self.initialization["robot_q"]
            self.views["qpos"][ids] = torch.as_tensor(qpos, dtype=torch.float32, device=self.device)
            self.views["qvel"][ids] = 0
            self.tick[ids] = 0
            self.views["time"][ids] = 0
            return
        indices = self.rng.choice(len(self.rsi_probability), len(ids), p=self.rsi_probability)
        for key in ("qpos", "qvel"):
            self.views[key][ids] = torch.tensor(self.rsi[key][indices], dtype=torch.float32, device=self.device)
        self.tick[ids] = torch.tensor(self.rsi_ticks[indices], dtype=torch.long, device=self.device)
        self.views["time"][ids] = self.tick[ids].float()*PHYSICS_DT_S

    def reward(self, state, target, power, actions):
        return residual_reward(self.robot, self.part_indices, state, target, power, actions,
                               self.completed_controls, self.scale, self.contact_history)

    def sampler_state(self):
        return dict(rsi=self.rng.bit_generator.state,
                    curriculum=self.curriculum.state_dict() if self.curriculum else None)

    def load_sampler_state(self, state):
        self.rng.bit_generator.state = state["rsi"]
        if self.curriculum is not None:
            self.curriculum.load_state_dict(state["curriculum"])
