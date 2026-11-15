"""Position targets only. Servo gains and all physical limits belong to the task."""

import torch


CONTROL = dict(schema="virtual-wrist-position-v3", wrist_translation_m=.12, wrist_rotation_rad=1., moving_average=1.)


class PositionControl:
    def __init__(self, robot, count, device, config):
        self.robot = robot
        self.lower = torch.tensor(robot["lower"], device=device)
        self.upper = torch.tensor(robot["upper"], device=device)
        self.previous = torch.zeros(count, len(robot["joint_names"]), device=device)
        self.wrist_scale = torch.tensor([config["wrist_translation_m"]]*3 + [config["wrist_rotation_rad"]]*3, device=device)
        if config.get("schema") != CONTROL["schema"]:
            raise ValueError(f"Unsupported position control schema: {config.get('schema')}")
        self.ctrl_lower = torch.tensor(robot["command_lower"], device=device)
        self.ctrl_upper = torch.tensor(robot["command_upper"], device=device)
        if torch.any(self.wrist_scale <= 0):
            raise ValueError("Invalid position control scales")
        if config["moving_average"] != 1:
            raise ValueError("Current-position wrist control requires moving_average=1")

    def reset(self, indices, positions):
        self.previous[indices] = positions

    def decode(self, actions, sides, residual, positions):
        """Targets from one soft-clamped action and the joint positions measured at this control step."""
        sizes = [len(self.robot["hands"][s]["indices"]) for s in sides]
        width = sum(sizes)
        if actions.shape[1] != width*(2 if residual else 1):
            raise ValueError("Action dimensions differ from position-control contract")
        # Native PPO/player applies soft-clamp before this boundary.
        base = actions[:, :width]
        offset = actions[:, width:]*2 if residual else torch.zeros_like(base)
        origin = positions
        if origin is None or origin.shape != self.previous.shape or not torch.isfinite(origin).all():
            raise ValueError("Missing or invalid wrist origin; measured joint positions are required")
        result = origin.clone()
        start = 0
        for side, size in zip(sides, sizes):
            ids = self.robot["hands"][side]["indices"]
            hand = base[:, start:start+size] + offset[:, start:start+size]
            result[:, ids[:6]] = origin[:, ids[:6]] + hand[:, :6]*self.wrist_scale
            finger = hand[:, 6:].clamp(-1, 1)
            result[:, ids[6:]] = self.lower[ids[6:]] + (finger+1)*0.5*(self.upper[ids[6:]]-self.lower[ids[6:]])
            start += size
        self.previous = torch.maximum(torch.minimum(result, self.ctrl_upper), self.ctrl_lower)
        if not torch.isfinite(self.previous).all():
            raise ValueError("Non-finite position target")
        return self.previous
