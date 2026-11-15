"""Shared training/deployment features. All physical inputs use bench conventions."""

import torch


def conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def multiply(a, b):
    w = a[..., :1]*b[..., :1] - (a[..., 1:]*b[..., 1:]).sum(-1, keepdim=True)
    v = a[..., :1]*b[..., 1:] + b[..., :1]*a[..., 1:] + torch.cross(a[..., 1:], b[..., 1:], dim=-1)
    return torch.cat((w, v), dim=-1)


def rotate(q, v):
    t = 2 * torch.cross(q[..., 1:], v, dim=-1)
    return v + q[..., :1]*t + torch.cross(q[..., 1:], t, dim=-1)


def slerp(a, b, fraction):
    dot = (a*b).sum(-1, keepdim=True)
    b = torch.where(dot < 0, -b, b)
    dot = dot.abs().clamp(0, 1)
    theta = torch.acos(dot)
    denominator = torch.sin(theta).clamp_min(1e-8)
    curved = (torch.sin((1-fraction)*theta)*a + torch.sin(fraction*theta)*b)/denominator
    linear = a + fraction*(b-a)
    return torch.nn.functional.normalize(torch.where(dot > 0.9995, linear, curved), dim=-1)


def xyzw(q):
    return q[..., [1, 2, 3, 0]]


def state_difference(target, actual):
    rotation = multiply(target[..., 3:7], conjugate(actual[..., 3:7]))
    rotation = torch.where(rotation[..., :1] < 0, -rotation, rotation)
    return torch.cat((target[..., :3]-actual[..., :3], rotation,
                      target[..., 7:]-actual[..., 7:]), -1)


class Reference:
    def __init__(self, arrays, device):
        self.values = {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in arrays.items()}
        self.times = self.values["time_s"]

    def sample(self, times):
        times = times.clamp(self.times[0], self.times[-1])
        upper = torch.searchsorted(self.times, times.contiguous()).clamp(1, len(self.times)-1)
        lower = upper-1
        weight = (times-self.times[lower])/(self.times[upper]-self.times[lower])
        result = {}
        for key, values in self.values.items():
            if key.endswith(("_vertices", "_bps")):
                result[key] = values
                continue
            a, b = values[lower], values[upper]
            f = weight.reshape(-1, *([1]*(a.ndim-1)))
            value = a + f*(b-a)
            if key == "object_state" or key.endswith("_wrist"):
                value[..., 3:7] = slerp(a[..., 3:7], b[..., 3:7], f)
            result[key] = value
        return result


class Features:
    def __init__(self, robot, part_names, gravity=9.81):
        self.robot, self.parts, self.gravity = robot, part_names, gravity
        self.base_dimensions = None

    def hand(self, state, target, side, stage):
        hand = self.robot["hands"][side]
        ids = hand["indices"]
        q, dq = state["robot_q"][:, ids], state["robot_dq"][:, ids]
        bodies = torch.stack([state["bodies"][b] for b in hand["bodies"]], dim=1)
        wrist = bodies[:, 0]
        goal = target[f"{side}_wrist"]
        joints = target[f"{side}_human"][:, 1:]
        joint_vel = target[f"{side}_human_velocity"][:, 1:]
        prop_wrist = torch.cat([torch.zeros_like(wrist[:, :3]), xyzw(wrist[:, 3:7]), wrist[:, 7:]], dim=-1)
        obs = {
            "privileged": torch.cat([dq[:, 6:], dq[:, :6]], -1),
            "proprioception": torch.cat([q[:, 6:], q[:, 6:].cos(), q[:, 6:].sin(), prop_wrist, q[:, :6]], -1),
            "target": torch.cat([
                goal[:, :3]-wrist[:, :3], goal[:, 7:10], goal[:, 7:10]-wrist[:, 7:10],
                xyzw(goal[:, 3:7]), xyzw(multiply(wrist[:, 3:7], conjugate(goal[:, 3:7]))),
                goal[:, 10:13], goal[:, 10:13]-wrist[:, 10:13],
                (joints-bodies[:, 1:, :3]).flatten(1), joint_vel.flatten(1),
                (joint_vel-bodies[:, 1:, 7:10]).flatten(1), target["robot_q"][:, ids[:6]],
            ], -1),
        }
        self.base_dimensions = {key: value.shape[-1] for key, value in obs.items()}
        if stage == "imitator":
            return obs
        object_index = self.parts.index(hand["object_part"])
        actual = state["objects"][:, object_index]
        expected = target["object_state"][:, object_index]
        force = state["tip_force"][side]
        force4 = torch.cat((force, force.norm(dim=-1, keepdim=True)), -1).flatten(1)
        com_local = torch.as_tensor(hand["object_com"], dtype=q.dtype, device=q.device).expand(q.shape[0], -1)
        com = actual[:, :3] + rotate(actual[:, 3:7], com_local) - wrist[:, :3]
        weight = torch.full_like(q[:, :1], hand["object_mass"]*self.gravity)
        extra_priv = torch.cat([actual[:, :3]-wrist[:, :3], xyzw(actual[:, 3:7]), actual[:, 7:13], force4, com, weight,
                                state["objects"].flatten(1), state["object_q"], state["object_dq"]], -1)
        bps = target[f"{side}_bps"].expand(q.shape[0], -1)
        extra_target = torch.cat([
            expected[:, :3]-actual[:, :3], expected[:, 7:10], expected[:, 7:10]-actual[:, 7:10],
            xyzw(expected[:, 3:7]), xyzw(multiply(actual[:, 3:7], conjugate(expected[:, 3:7]))),
            expected[:, 10:13], expected[:, 10:13]-actual[:, 10:13],
            (actual[:, None, :3]-bodies[:, :, :3]).norm(dim=-1), target[f"{side}_tip_distance"], bps,
            state_difference(target["object_state"], state["objects"]).flatten(1), target["object_q"]-state["object_q"],
        ], -1)
        obs["privileged"] = torch.cat([obs["privileged"], extra_priv], -1)
        obs["target"] = torch.cat([obs["target"], extra_target], -1)
        return obs

    def __call__(self, state, target, sides, stage):
        values = [self.hand(state, target, side, stage) for side in sides]
        return {key: torch.cat([v[key] for v in values], -1).clamp(-5, 5) for key in values[0]}


def observed_bodies(robot, object_bodies):
    """Every body the features read: the tracked hand links, then the object parts."""
    return list(dict.fromkeys([b for h in robot["hands"].values() for b in h["bodies"]] + list(object_bodies)))


def protocol_state(frames, robot, object_bodies, orders, device):
    """Evaluation frames as the training state. `orders` maps the method's robot and object joint
    order into the announced one; contact records become the net force on each fingertip."""
    def tensor(values):
        return torch.as_tensor(values, dtype=torch.float32, device=device)

    state = {key: tensor([f[key] for f in frames])[:, orders[key.split("_")[0]]]
             for key in ("robot_q", "robot_dq", "object_q", "object_dq")}
    state["bodies"] = {b: tensor([f["bodies"][b] for f in frames]) for b in observed_bodies(robot, object_bodies)}
    state["objects"] = torch.stack([state["bodies"][b] for b in object_bodies], dim=1)
    state["tip_force"] = {s: torch.zeros((len(frames), 5, 3), device=device) for s in robot["sides"]}
    lookup = {b: (s, i) for s in robot["sides"] for i, b in enumerate(robot["hands"][s]["contact_bodies"])}
    for row, frame in enumerate(frames):
        if frame.get("contacts") is None:
            raise ValueError("Required contact observations are missing")
        for contact in frame["contacts"]["records"]:
            for field, sign in (("body1", -1), ("body2", 1)):
                if contact[field] in lookup:
                    side, tip = lookup[contact[field]]
                    state["tip_force"][side][row, tip] += sign*tensor(contact["force_on_body2_world"])
    return state
