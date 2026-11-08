"""Object-free, multi-recording single-hand pretraining with native RSI."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from .hand_reference import WristReference
from .io import load_artifact, read_json, sha256
from .mjlab_env import MjlabVecEnv, hand_model
from .motion import MotionCache
from .residual_reward import hand_reward
from .robot_physics import require_compatibility


class ImitatorEnv(MjlabVecEnv):
    def __init__(self, prepared, count, device, seed=42, allow_partial=False):
        self.path, metadata = load_artifact(prepared)
        if metadata["format"] != "maniptrans-h2r-pretrain-v1":
            raise ValueError("Expected generalist pretraining reference")
        if metadata["partial"] and not allow_partial:
            raise ValueError("Partial corpus requires --allow-incomplete")
        require_compatibility(metadata["compatibility"])
        robot = read_json(self.path / "robot_mapping.json")
        self.side = robot["sides"][0]
        model, bodies = hand_model(self.path / "robot_scene.xml", robot)
        self.motion = MotionCache(self.path, read_json(self.path / "motion_manifest.json"), count, self.side, device, seed, allow_partial)
        self.wrist = WristReference(model, robot, self.side)
        self.rng = np.random.default_rng(seed+1)
        self.diagnostic = False
        self.fixed_windows = None
        metadata = dict(metadata, scene_sha256=sha256(self.path / "robot_scene.xml"))
        super().__init__(model, robot, bodies, [], count, device, metadata, "imitator")
        self.finish_init()

    def target(self, lookahead=0):
        result = self.motion.sample(lookahead)
        q = torch.zeros((self.count, self.action_dim), device=self.device)
        q[:, self.robot["hands"][self.side]["indices"][:6]] = result.pop("wrist_q")
        result["robot_q"] = q
        return result

    def remaining_ticks(self):
        return torch.as_tensor(2*(self.motion.lengths-1-self.motion.positions), device=self.device)

    def advance_reference(self):
        self.motion.advance()

    def reset_states(self, ids):
        indices = ids.cpu().numpy()
        fixed = None if self.fixed_windows is None else np.asarray(self.fixed_windows)[indices]
        self.motion.reset(indices, fixed_windows=fixed, from_start=self.diagnostic)
        target = self.target(0)
        wrist = target[f"{self.side}_wrist"][ids].cpu().numpy()
        anchors = target["robot_q"][ids][:, self.robot["hands"][self.side]["indices"][3:6]].cpu().numpy()
        count = len(indices)
        lower, upper = np.asarray(self.robot["lower"]), np.asarray(self.robot["upper"])
        q = np.broadcast_to(np.clip(np.pi/36, lower, upper), (count, len(lower))).copy()
        dq = np.zeros_like(q)
        if not self.diagnostic:
            q[:, 6:] = np.clip(q[:, 6:]+self.rng.normal(size=q[:, 6:].shape)*(upper[6:]-lower[6:])/8, lower[6:], upper[6:])
            dq[:, 6:] = self.rng.normal(0, .1, size=q[:, 6:].shape)
        for i, reference in enumerate(wrist):
            for attempt in range(8):
                p, r = reference[:3].copy(), Rotation.from_quat(reference[3:7], scalar_first=True)
                if not self.diagnostic:
                    p += self.rng.normal(0, .01, 3)
                    axis = self.rng.random(3)
                    r = Rotation.from_rotvec(axis/np.linalg.norm(axis)*self.rng.normal(0, np.pi/18))*r
                try:
                    q[i, :6] = self.wrist.solve(p[None], r.as_matrix()[None], previous=anchors[i])[0]
                    break
                except ValueError:
                    if attempt == 7:
                        raise
        v, w = wrist[:, 7:10].copy(), wrist[:, 10:13].copy()
        if not self.diagnostic:
            v += self.rng.normal(0, .01, v.shape)
            w += self.rng.normal(0, .01, w.shape)
        dq[:, :6] = self.wrist.velocity(q[:, :6], v, w)
        self.views["qpos"][ids[:, None], torch.tensor(self.robot["qpos_indices"], device=self.device)] = torch.tensor(q, dtype=torch.float32, device=self.device)
        self.views["qvel"][ids[:, None], torch.tensor(self.robot["qvel_indices"], device=self.device)] = torch.tensor(dq, dtype=torch.float32, device=self.device)
        self.tick[ids] = torch.as_tensor(self.motion.positions[indices]*2, device=self.device)
        self.views["time"][ids] = self.tick[ids].float()/120

    def reward(self, state, target, power, actions):
        return hand_reward(self.robot, self.side, state, target, power, actions, self.elapsed_ticks, self.scale)

    def sampler_state(self):
        return dict(motion=self.motion.state_dict(), rsi=self.rng.bit_generator.state)

    def load_sampler_state(self, state):
        self.motion.load_state_dict(state["motion"])
        self.rng.bit_generator.state = state["rsi"]
