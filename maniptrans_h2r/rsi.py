"""Weighted robot IK and validated full-scene reference-state initialization."""

import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .robot import FIT_WEIGHTS, initial_qpos, object_roots


def fit(task_path, task, robot, arrays, initial, iterations, device):
    import pytorch_kinematics as pk
    import torch
    import yaml
    manifest_path = task_path / "scene/robot/assets/manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    path = manifest_path.parent / manifest["entrypoints"][task["robot_name"]]["urdf"]
    tree = ET.fromstring(path.read_bytes())
    for limit in tree.iter("limit"):
        limit.attrib.setdefault("effort", "0")
        limit.attrib.setdefault("velocity", "0")
    chain = pk.build_chain_from_urdf(ET.tostring(tree)).to(device=device, dtype=torch.float32)
    names = robot["joint_names"]
    parameters = chain.get_joint_parameter_names()
    if set(parameters) != set(names):
        raise ValueError("URDF differs from sealed robot")
    order = [names.index(n) for n in parameters]
    bodies = [b for s in robot["sides"] for b in robot["hands"][s]["bodies"]]
    model = mujoco.MjModel.from_xml_path(str(task_path / task["scene"]))
    data = mujoco.MjData(model)
    data.qpos[:] = initial_qpos(model, task)
    probe = np.random.default_rng(0).uniform(-.05, .05, len(names))
    data.qpos[robot["qpos_indices"]] = probe
    mujoco.mj_forward(model, data)
    transforms = chain.forward_kinematics(torch.tensor(probe[order][None], dtype=torch.float32, device=device))
    for body in bodies:
        if not np.allclose(transforms[body].get_matrix()[0, :3, 3].detach().cpu(), data.body(body).xpos, atol=2e-5):
            raise ValueError(f"URDF/task FK differs at {body}")
    wrist_ids = [i for s in robot["sides"] for i in robot["hands"][s]["indices"][:6]]
    finger_ids = [i for s in robot["sides"] for i in robot["hands"][s]["indices"][6:]]
    lower = torch.tensor(robot["lower"], device=device)
    upper = torch.tensor(robot["upper"], device=device)
    weights = torch.tensor(FIT_WEIGHTS*len(robot["sides"]), device=device)
    human = np.concatenate([arrays[f"{s}_human"] for s in robot["sides"]], axis=1)
    if iterations < 1:
        raise ValueError("IK iterations must be positive")
    started = time.monotonic()
    q0 = torch.tensor(initial, dtype=torch.float32, device=device)
    wrist = torch.nn.Parameter(q0[:, wrist_ids].clone())
    fingers = torch.nn.Parameter(q0[:, finger_ids].clone())
    optimizer = torch.optim.Adam([dict(params=[wrist], lr=.0008), dict(params=[fingers], lr=.0004)])
    target = torch.tensor(human, dtype=torch.float32, device=device)

    def chunk_loss(start, stop):
        q = q0[start:stop].clone()
        q[:, wrist_ids], q[:, finger_ids] = wrist[start:stop], fingers[start:stop]
        transforms = chain.forward_kinematics(q[:, order])
        points = torch.stack([transforms[b].get_matrix()[:, :3, 3] for b in bodies], 1)
        return ((points-target[start:stop]).norm(dim=-1)*weights).sum() / (len(q0)*len(bodies))

    stats = optimize_sequence(optimizer, chunk_loss, len(q0), iterations,
        lambda: (wrist.clamp_(lower[wrist_ids], upper[wrist_ids]),
                 fingers.clamp_(lower[finger_ids], upper[finger_ids])))
    q0[:, wrist_ids], q0[:, finger_ids] = wrist.detach(), fingers.detach()
    stats["elapsed_s"] = time.monotonic()-started
    return q0.cpu().numpy(), stats


def optimize_sequence(optimizer, chunk_loss, count, iterations, project, chunk_size=256):
    """One global mean, Adam update and native stop decision per iteration.

    chunk_loss returns its contribution to the full-sequence mean. Only the FK
    graph is chunked; parameter gradients and Adam state span the whole sequence.
    Like native fitting, the stop check uses the loss before the returned update.
    """
    import torch
    past_loss, reason = 1e10, "iteration_budget"
    for step in range(1, iterations+1):
        optimizer.zero_grad()
        losses = []
        for start in range(0, count, chunk_size):
            loss = chunk_loss(start, min(start+chunk_size, count))
            loss.backward()
            losses.append(loss.detach())
        value = float(torch.stack(losses).sum())
        if not np.isfinite(value):
            raise FloatingPointError("Non-finite IK loss")
        optimizer.step()
        with torch.no_grad():
            project()
        if step % 100 == 0:
            print(f"IK step {step}/{iterations}: {value:.6f}", flush=True)
            if past_loss-value < 1e-5:
                reason = "native_loss_improvement"
                break
            past_loss = value
    print(f"IK stopped at {step}: {reason}, loss={value:.6f}", flush=True)
    return dict(iterations=step, iteration_limit=iterations, stop_reason=reason,
                loss=value, loss_timing="before_final_update", output_timing="after_final_update")


def states(model, task, robot, reference, optimized):
    from .hand_reference import WristReference
    times = reference["time_s"]
    for side in robot["sides"]:
        wrist = WristReference(model, robot, side)
        data = mujoco.MjData(model)
        poses, rotations = [], []
        for q in optimized:
            data.qpos[robot["qpos_indices"]] = q
            mujoco.mj_forward(model, data)
            poses.append(data.xpos[wrist.body].copy())
            rotations.append(data.xmat[wrist.body].reshape(3, 3).copy())
        optimized[:, robot["hands"][side]["indices"][:6]] = wrist.solve(np.asarray(poses), np.asarray(rotations))
    qpos = np.broadcast_to(initial_qpos(model, task), (len(times), model.nq)).copy()
    qpos[:, robot["qpos_indices"]] = optimized
    parts = list(task["object"]["parts"])
    for root in object_roots(task).values():
        if root["joint"] is None:
            continue
        names = [p for p, b in task["object"]["parts"].items() if b == root["body"]]
        if len(names) != 1:
            raise ValueError("Object root must have a named reference part")
        index = parts.index(names[0])
        joint = model.joint(root["joint"])
        adr = int(joint.qposadr[0])
        qpos[:, adr:adr+7] = reference["object_state"][:, index, :7]
        for i in range(1, len(times)):
            if np.dot(qpos[i-1, adr+3:adr+7], qpos[i, adr+3:adr+7]) < 0:
                qpos[i, adr+3:adr+7] *= -1
    for j, name in enumerate(task["object"]["joints"].values()):
        qpos[:, int(model.joint(name).qposadr[0])] = reference["object_q"][:, j]
    qvel = np.empty((len(times), model.nv))
    for i in range(len(times)):
        lo, hi = max(i-1, 0), min(i+1, len(times)-1)
        mujoco.mj_differentiatePos(model, qvel[i], times[hi]-times[lo], qpos[lo], qpos[hi])
    qvel = gaussian_filter1d(qvel, 2, axis=0, mode="nearest")
    return qpos, qvel


def validate(model, task, robot, reference, qpos, surfaces):
    data = mujoco.MjData(model)
    valid, reasons = np.ones(len(qpos), dtype=bool), []
    max_penetration = task["initialization"]["max_penetration_m"]
    for i, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        problems = []
        if not np.isfinite(pose).all():
            problems.append("nonfinite")
        if any(-float(c.dist) > max_penetration for c in data.contact):
            problems.append("penetration")
        for j in range(model.njnt):
            if model.jnt_limited[j]:
                q = pose[int(model.jnt_qposadr[j])]
                if q < model.jnt_range[j, 0]-1e-7 or q > model.jnt_range[j, 1]+1e-7:
                    problems.append("joint_limit")
        world = []
        for part, vertices in surfaces.items():
            part_index = list(task["object"]["parts"]).index(part)
            obj = reference["object_state"][i, part_index]
            world.append(vertices@Rotation.from_quat(obj[3:7], scalar_first=True).as_matrix().T+obj[:3])
        tree = cKDTree(np.concatenate(world))
        for side in robot["sides"]:
            tips = [robot["hands"][side]["bodies"][k] for k in (4, 8, 12, 16, 20)]
            distance = tree.query(np.stack([data.body(b).xpos for b in tips]))[0]
            near = reference[f"{side}_tip_distance"][i] < .03
            if np.any(near & (distance > .03)):
                problems.append(f"{side}_not_near_reference_contact")
        reasons.append(problems)
        valid[i] = not problems
    # RSI starts only at control boundaries and must leave a nonempty suffix.
    ticks = np.rint(reference["time_s"]*120).astype(np.int64)
    valid &= (ticks % 2 == 0) & (ticks < task["end_tick"])
    return valid, reasons


def sampling_probabilities(count, valid):
    """Exact masses of floor(.98*N*U), conditioned on valid original indices."""
    indices = np.arange(count)
    mass = np.maximum(0, np.minimum(indices+1, .98*count)-indices)/(.98*count)
    mass[-2] += mass[-1]  # Clamp the original law to at least one transition.
    mass[-1] = 0
    mass *= valid
    if mass.sum() == 0:
        raise ValueError("no_valid_rsi: no valid initialization in the native sampling support")
    return mass/mass.sum()
