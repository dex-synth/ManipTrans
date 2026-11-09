"""Canonical hand targets, MANO interpolation and continuous virtual-wrist IK."""

from contextlib import contextmanager
from functools import lru_cache
import inspect
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


def lift_euler(rotations, previous=None):
    """Lift SO(3) continuously without using joint limits to wrap a path."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        eulers = Rotation.from_matrix(rotations).as_euler("XYZ")
    result = []
    anchor = np.zeros(3) if previous is None else np.asarray(previous)
    last = anchor.copy()
    for matrix, angles in zip(rotations, eulers):
        bases = [angles, np.array([angles[0]+np.pi, np.pi-angles[1], angles[2]+np.pi])]
        if abs(np.cos(angles[1])) < 1e-7:
            bases.append(np.array([angles[0]-np.sign(angles[1])*last[2], angles[1], last[2]]))
        candidates = np.asarray(bases)
        candidates += 2*np.pi*np.rint((last-candidates)/(2*np.pi))
        last = candidates[np.argmin(np.linalg.norm(candidates-last, axis=1))]
        if not np.allclose(Rotation.from_euler("XYZ", last).as_matrix(), matrix, atol=1e-5):
            raise ValueError("Wrist Euler inverse failed FK check")
        result.append(last.copy())
    return np.asarray(result)


def bounded_euler(path, lower, upper):
    """Choose one constant equivalent branch/turn offset, or return None."""
    legal = []
    for candidate in (path, path*np.array([1, -1, 1])+np.pi):
        lo = np.ceil((np.asarray(lower)-candidate.min(axis=0)-1e-8)/(2*np.pi))
        hi = np.floor((np.asarray(upper)-candidate.max(axis=0)+1e-8)/(2*np.pi))
        if np.all(lo <= hi):
            shift = np.clip(np.rint(-candidate[0]/(2*np.pi)), lo, hi)
            legal.append(candidate+2*np.pi*shift)
    return min(legal, key=lambda q: np.linalg.norm(q[0])) if legal else None


def continuous_euler(rotations, lower, upper, previous=None):
    """Fit a complete path, or anchor RSI to its existing window branch."""
    result = lift_euler(rotations, previous)
    if previous is None:
        result = bounded_euler(result, lower, upper)
        if result is None:
            raise ValueError("robot_reference_invalid: no legal continuous wrist rotation branch")
    if np.any(result < np.asarray(lower)-1e-7) or np.any(result > np.asarray(upper)+1e-7):
        raise ValueError("robot_reference_invalid: anchored wrist rotation branch outside limits")
    return result


def wrist_windows(path, lower, upper, max_samples=2000):
    """Cover every transition once; branch changes occur only between episodes.

    Feasibility of a prefix is monotone, so binary search finds its longest
    legal end. A short final window is balanced when both resulting paths fit.
    Short windows forced by limits are retained, never discarded or padded.
    """
    if len(path) < 2 or max_samples < 2:
        raise ValueError("A wrist window needs at least one transition")
    def fit(a, b):
        return bounded_euler(path[a:b], lower, upper)
    spans, start = [], 0
    while start < len(path)-1:
        low, high = start+2, min(start+max_samples, len(path))
        if fit(start, low) is None:
            raise ValueError(f"robot_reference_invalid: wrist transition {start}:{low} cannot fit limits")
        while low < high:
            middle = (low+high+1)//2
            if fit(start, middle) is None:
                high = middle-1
            else:
                low = middle
        spans.append((start, low))
        start = low-1
    if len(spans) > 1 and spans[-1][1]-spans[-1][0] < 31:
        a, b = spans[-2][0], spans[-1][1]
        candidates = range(max(a+30, b-max_samples), min(b-30, a+max_samples))
        for boundary in sorted(candidates, key=lambda i: abs(2*i-a-b+1)):
            if fit(a, boundary+1) is not None and fit(boundary, b) is not None:
                spans[-2:] = [(a, boundary+1), (boundary, b)]
                break
    return [(a, fit(a, b)) for a, b in spans]


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
        translation = self.translation(positions)
        relative = self.parent_rotation.T @ rotations @ self.local_rotation.T
        angles = continuous_euler(relative, self.lower[3:], self.upper[3:], previous)
        result = np.concatenate([translation, angles], axis=1)
        if np.any(result < self.lower-1e-7) or np.any(result > self.upper+1e-7):
            raise ValueError("robot_reference_invalid: wrist outside sealed workspace")
        return result

    def translation(self, positions):
        q = np.linalg.solve(self.translation_axes, (positions-self.origin).T).T
        if np.any(q < self.lower[:3]-1e-7) or np.any(q > self.upper[:3]+1e-7):
            raise ValueError("robot_reference_invalid: wrist outside sealed workspace")
        return q

    def rotations_from_points(self, human):
        palm = [5, 9, 13, 17]
        source = self.template[palm]-self.template[:1]
        desired = human[:, palm]-human[:, :1]
        u, _, vt = np.linalg.svd(np.einsum("ni,tnj->tij", source, desired))
        correction = np.broadcast_to(np.eye(3), (len(human), 3, 3)).copy()
        correction[:, 2, 2] = np.linalg.det(vt.swapaxes(-1, -2)@u.swapaxes(-1, -2))
        return vt.swapaxes(-1, -2)@correction@u.swapaxes(-1, -2)@self.zero_rotation

    def from_points(self, human):
        rotation = self.rotations_from_points(human)
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
    return reference_values(human, q, quat, times, side)


def pretraining_reference(model, robot, side, points, times):
    """Compute physical targets/derivatives once, then fit per-episode angles."""
    human = np.asarray(points)[:, MANO_ORDER]
    wrist = WristReference(model, robot, side)
    rotations = wrist.rotations_from_points(human)
    angles = lift_euler(wrist.parent_rotation.T @ rotations @ wrist.local_rotation.T)
    segments = wrist_windows(angles, wrist.lower[3:], wrist.upper[3:])
    q = np.concatenate([wrist.translation(human[:, 0]), angles], axis=1)
    quat = quaternion_signs(Rotation.from_matrix(rotations).as_quat(scalar_first=True))
    # Unbounded angles are replaced by each segment's legal angles before
    # storage. World targets and velocities retain their whole-clip values.
    return reference_values(human, q, quat, times, side), segments


def reference_values(human, q, quat, times, side):
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


def interpolation_times(times, source_times):
    """Snap sub-nanosecond endpoint roundoff, but reject real extrapolation."""
    times = np.asarray(times, dtype=np.float64)
    lower, upper = source_times[0], source_times[-1]
    if not np.isfinite(times).all() or np.any(times < lower-1e-9) or np.any(times > upper+1e-9):
        raise ValueError(f"Interpolation times outside source interval [{lower}, {upper}]")
    return np.clip(times, lower, upper)


def interpolate_motion(arrays, metadata, times, mano_dir, *, endpoints=None, mirror=False):
    """SLERP absolute MANO rotations, decode, then anchor to the real wrist."""
    import torch
    source_times = arrays["time_s"]
    source_quat = np.concatenate([arrays["wrist_quat_wxyz"][:, None], arrays["local_quat_wxyz"]], axis=1)
    positions = arrays["joints_m"][:, 0]
    if endpoints is not None:
        # First query the selected original poses, then interpolate between them.
        endpoints = interpolation_times(endpoints, source_times)
        position = np.stack([np.interp(endpoints, source_times, positions[:, j]) for j in range(3)], axis=-1)
        quat = np.stack([Slerp(source_times, Rotation.from_quat(source_quat[:, j], scalar_first=True))(endpoints).as_quat(scalar_first=True)
                         for j in range(16)], axis=1)
        source_times, source_quat, positions = np.array([0., endpoints[1]-endpoints[0]]), quat, position
    times = interpolation_times(times, source_times)
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
