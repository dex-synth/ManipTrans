"""Canonical hand targets, MANO interpolation and continuous virtual-wrist IK."""

from contextlib import contextmanager
from functools import lru_cache
import inspect
import itertools
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp

from .robot import MANO_ORDER

CONTROL_DT = 1/60


def quaternion_signs(quat):
    result = np.asarray(quat, dtype=np.float64).copy()
    result /= np.linalg.norm(result, axis=-1, keepdims=True)
    result[0] *= np.where(result[0, ..., :1] < 0, -1, 1)
    for i in range(1, len(result)):
        result[i] *= np.where((result[i-1]*result[i]).sum(-1, keepdims=True) < 0, -1, 1)
    return result


def velocities(points, quat, times):
    linear = gaussian_filter1d(np.gradient(points, times, axis=0), 2, axis=0, mode="nearest")
    rot = Rotation.from_quat(quat, scalar_first=True)
    angular = (rot[1:]*rot[:-1].inv()).as_rotvec()/np.diff(times)[:, None]
    angular = gaussian_filter1d(np.concatenate([angular, angular[-1:]]), 2, axis=0, mode="nearest")
    return linear, angular


def continuous_euler(rotations, lower, upper, previous=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        eulers = Rotation.from_matrix(rotations).as_euler("XYZ")
    result = []
    previous = np.zeros(3) if previous is None else np.asarray(previous)
    shifts = np.asarray(list(itertools.product((-1, 0, 1), repeat=3)))*2*np.pi
    for matrix, angles in zip(rotations, eulers):
        bases = [angles, np.array([angles[0]+np.pi, np.pi-angles[1], angles[2]+np.pi])]
        if abs(np.cos(angles[1])) < 1e-7:
            bases.append(np.array([angles[0]-np.sign(angles[1])*previous[2], angles[1], previous[2]]))
        candidates = (np.asarray(bases)[:, None] + shifts).reshape(-1, 3)
        candidates = candidates[((candidates >= lower-1e-8) & (candidates <= upper+1e-8)).all(axis=1)]
        if len(candidates) == 0:
            raise ValueError("robot_reference_invalid: no legal wrist rotation branch")
        previous = candidates[np.argmin(np.linalg.norm(candidates-previous, axis=1))]
        if not np.allclose(Rotation.from_euler("XYZ", previous).as_matrix(), matrix, atol=1e-5):
            raise ValueError("Wrist Euler inverse failed FK check")
        result.append(previous.copy())
    return np.asarray(result)


class WristReference:
    def __init__(self, model, robot, side):
        import mujoco
        self.model, self.robot, self.side = model, robot, side
        self.hand = robot["hands"][side]
        self.ids = self.hand["indices"][:6]
        self.qids = np.asarray(robot["qpos_indices"])[self.ids]
        self.dids = np.asarray(robot["qvel_indices"])[self.ids]
        self.lower, self.upper = np.asarray(robot["lower"])[self.ids], np.asarray(robot["upper"])[self.ids]
        data = mujoco.MjData(model)
        data.qpos[robot["qpos_indices"]] = np.clip(0, robot["lower"], robot["upper"])
        mujoco.mj_forward(model, data)
        self.body = model.body(self.hand["bodies"][0]).id
        self.origin = data.xpos[self.body].copy()
        self.template = np.stack([data.body(b).xpos.copy() for b in self.hand["bodies"]])
        self.zero_rotation = data.xmat[self.body].reshape(3, 3).copy()
        roll_body = int(model.joint(self.hand["joints"][3]).bodyid[0])
        self.parent_rotation = data.xmat[int(model.body_parentid[roll_body])].reshape(3, 3).copy()
        self.local_rotation = self.parent_rotation.T @ self.zero_rotation
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jp, jr, self.body)
        self.translation_axes = jp[:, self.dids[:3]].copy()
        if not np.allclose(jp[:, self.dids[3:]], 0, atol=1e-8):
            raise ValueError("Unsupported wrist offset from rotational joint origin")

    def solve(self, positions, rotations, previous=None):
        translation = np.linalg.solve(self.translation_axes, (positions-self.origin).T).T
        relative = self.parent_rotation.T @ rotations @ self.local_rotation.T
        angles = continuous_euler(relative, self.lower[3:], self.upper[3:], previous)
        result = np.concatenate([translation, angles], axis=1)
        if np.any(result < self.lower-1e-7) or np.any(result > self.upper+1e-7):
            raise ValueError("robot_reference_invalid: wrist outside sealed workspace")
        return result

    def from_points(self, human):
        palm = [5, 9, 13, 17]
        source = self.template[palm]-self.template[:1]
        desired = human[:, palm]-human[:, :1]
        u, _, vt = np.linalg.svd(np.einsum("ni,tnj->tij", source, desired))
        correction = np.broadcast_to(np.eye(3), (len(human), 3, 3)).copy()
        correction[:, 2, 2] = np.linalg.det(vt.swapaxes(-1, -2)@u.swapaxes(-1, -2))
        rotation = vt.swapaxes(-1, -2)@correction@u.swapaxes(-1, -2)@self.zero_rotation
        q = self.solve(human[:, 0], rotation)
        return q, quaternion_signs(Rotation.from_matrix(rotation).as_quat(scalar_first=True))

    def velocity(self, q, linear, angular):
        import mujoco
        data = mujoco.MjData(self.model)
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        result = []
        for pose, v, w in zip(q, linear, angular):
            data.qpos[self.qids] = pose
            mujoco.mj_forward(self.model, data)
            mujoco.mj_jacBody(self.model, data, jp, jr, self.body)
            jac = np.concatenate([jp[:, self.dids], jr[:, self.dids]], axis=0)
            wanted = np.concatenate([v, w])
            dq = np.linalg.pinv(jac, rcond=1e-6)@wanted
            if not np.allclose(jac@dq, wanted, rtol=0, atol=1e-4):
                raise ValueError("Wrist Jacobian cannot realize reference velocity")
            result.append(dq)
        return np.asarray(result)


def reference_from_points(model, robot, side, points, times):
    human = np.asarray(points)[:, MANO_ORDER]
    wrist = WristReference(model, robot, side)
    q, quat = wrist.from_points(human)
    linear, angular = velocities(human[:, 0], quat, times)
    return {f"{side}_human": human.astype(np.float32),
            f"{side}_human_velocity": gaussian_filter1d(np.gradient(human, times, axis=0), 2, axis=0, mode="nearest").astype(np.float32),
            f"{side}_wrist": np.concatenate([human[:, 0], quat, linear, angular], axis=1).astype(np.float32),
            "wrist_q": q.astype(np.float32)}


@contextmanager
def legacy_mano():
    aliases = dict(bool=bool, int=int, float=float, complex=complex, object=object, unicode=str, str=str)
    added = {k: v for k, v in aliases.items() if k not in np.__dict__}
    old = getattr(inspect, "getargspec", None)
    try:
        for k, v in added.items():
            setattr(np, k, v)
        if old is None:
            inspect.getargspec = inspect.getfullargspec
        yield
    finally:
        for k in added:
            delattr(np, k)
        if old is None:
            delattr(inspect, "getargspec")


@lru_cache(maxsize=4)
def mano_model(directory, side, fix):
    import smplx
    import torch
    with legacy_mano():
        model = smplx.MANO(directory, is_rhand=side == "right", use_pca=False,
                           flat_hand_mean=True, create_transl=False).eval()
    if fix and side == "left":
        right = mano_model(directory, "right", False)
        if torch.sum(torch.abs(model.shapedirs[:, 0]-right.shapedirs[:, 0])) < 1:
            model.shapedirs[:, 0] *= -1
    return model


def interpolate_motion(arrays, metadata, times, mano_dir, *, endpoints=None, mirror=False):
    """SLERP absolute MANO rotations, decode, then anchor to the real wrist."""
    import torch
    source_times = arrays["time_s"]
    source_quat = np.concatenate([arrays["wrist_quat_wxyz"][:, None], arrays["local_quat_wxyz"]], axis=1)
    positions = arrays["joints_m"][:, 0]
    if endpoints is not None:
        # First query the selected original poses, then interpolate between them.
        position = np.stack([np.interp(endpoints, source_times, positions[:, j]) for j in range(3)], axis=-1)
        quat = np.stack([Slerp(source_times, Rotation.from_quat(source_quat[:, j], scalar_first=True))(endpoints).as_quat(scalar_first=True)
                         for j in range(16)], axis=1)
        source_times, source_quat, positions = np.array([0., endpoints[1]-endpoints[0]]), quat, position
    quat = np.stack([Slerp(source_times, Rotation.from_quat(source_quat[:, j], scalar_first=True))(times).as_rotvec()
                     for j in range(16)], axis=1).reshape(-1, 48)
    wrist = np.stack([np.interp(times, source_times, positions[:, j]) for j in range(3)], axis=-1)
    model = mano_model(str(mano_dir), metadata["source_side"], metadata["hot3d_shapedirs_fix"])
    result = []
    with torch.inference_mode():
        for start in range(0, len(times), 256):
            angles = torch.tensor(quat[start:start+256], dtype=torch.float32)
            out = model(global_orient=angles[:, :3], hand_pose=angles[:, 3:],
                        betas=torch.tensor(np.broadcast_to(arrays["betas"], (len(angles), 10)).copy(), dtype=torch.float32))
            joints = torch.cat([out.joints[:, :16], out.vertices[:, metadata["tip_vertex_ids"]]], dim=1)
            result.append((joints-joints[:, :1]).numpy()+wrist[start:start+256, None])
    result = np.concatenate(result)
    if mirror:
        result[..., 0] *= -1
    return result.astype(np.float32)
