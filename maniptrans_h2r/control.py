"""Position targets only. Servo gains and all physical limits belong to the task."""

import torch


class PositionControl:
    def __init__(self, robot, count, device, config):
        self.robot = robot
        self.lower = torch.tensor(robot["lower"], device=device)
        self.upper = torch.tensor(robot["upper"], device=device)
        self.previous = torch.zeros(count, len(robot["joint_names"]), device=device)
        self.wrist_scale = torch.tensor([config["wrist_translation_m"]]*3 + [config["wrist_rotation_rad"]]*3, device=device)
        self.alpha = config["moving_average"]
        self.limits = config.get("schema") == "virtual-wrist-position-v2"
        if self.limits:
            self.ctrl_lower = torch.tensor(robot["command_lower"], device=device)
            self.ctrl_upper = torch.tensor(robot["command_upper"], device=device)
        if not 0 < self.alpha <= 1 or torch.any(self.wrist_scale <= 0):
            raise ValueError("Invalid position control scales or moving average")

    def reset(self, indices, positions):
        self.previous[indices] = positions

    def decode(self, actions, reference, sides, residual):
        sizes = [len(self.robot["hands"][s]["indices"]) for s in sides]
        width = sum(sizes)
        if actions.shape[1] != width*(2 if residual else 1):
            raise ValueError("Action dimensions differ from position-control contract")
        # Native PPO/player applies soft-clamp before this boundary.
        base = actions[:, :width]
        offset = actions[:, width:]*2 if residual else torch.zeros_like(base)
        result = reference.clone()  # Inactive hand during single-hand imitator training follows method IK.
        start = 0
        for side, size in zip(sides, sizes):
            ids = self.robot["hands"][side]["indices"]
            hand = base[:, start:start+size] + offset[:, start:start+size]
            result[:, ids[:6]] = reference[:, ids[:6]] + hand[:, :6]*self.wrist_scale
            finger = hand[:, 6:].clamp(-1, 1)
            result[:, ids[6:]] = self.lower[ids[6:]] + (finger+1)*0.5*(self.upper[ids[6:]]-self.lower[ids[6:]])
            start += size
        self.previous = self.alpha*result + (1-self.alpha)*self.previous
        if self.limits:
            self.previous = torch.maximum(torch.minimum(self.previous, self.ctrl_upper), self.ctrl_lower)
        if not torch.isfinite(self.previous).all():
            raise ValueError("Non-finite position target")
        return self.previous
