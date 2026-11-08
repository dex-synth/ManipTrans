"""Complete task targets and independently optimized reference-state resets."""

from pathlib import Path
import shutil

import mujoco
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from .geometry import encode_bps
from .hand_reference import WristReference, reference_from_points
from .io import check_inputs, load_arrays, load_task, read_json, seal, write_json
from .prepare import body_reference, sample_mesh, velocity
from .robot import command_limits, mapping
from .robot_physics import require_clock, require_contacts, signature
from . import rsi


def resample(values, source_times, times):
    shape = values.shape[1:]
    flat = values.reshape(len(values), -1)
    return np.stack([np.interp(times, source_times, flat[:, i]) for i in range(flat.shape[1])], -1).reshape(len(times), *shape)


def prepare(args):
    import torch
    torch.set_num_threads(4)
    source, task = load_task(args.task)
    require_clock(task)
    require_contacts(task)
    if args.bundle or args.robot_bundle:
        if not args.bundle or not args.robot_bundle:
            raise ValueError("Pass both --bundle and --robot-bundle")
        check_inputs(args.bundle, args.robot_bundle, source, task)
    output = Path(args.output).resolve()
    if output.is_relative_to(source):
        raise ValueError("Output must be outside the task")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source, output / "task")
    model = mujoco.MjModel.from_xml_path(str(source / task["scene"]))
    from .physics import verify_sealed
    verify_sealed(model, read_json(source / "scene/compiled.json"))
    robot = command_limits(model, mapping(model, task))
    raw = load_arrays(source / "reference.npz")
    common = read_json(source / "reference.json")
    ticks = np.unique(np.r_[np.arange(0, task["end_tick"]+1, 2), task["end_tick"]])
    times = ticks/120.
    if not np.isclose(raw["time_s"][-1], times[-1], rtol=0, atol=1e-10):
        raise ValueError("Task/reference endpoint mismatch")
    parts = list(task["object"]["parts"])
    scoring = task["object"].get("scoring_parts", parts)
    columns = [common["object_geometry"]["part_order"].index(n) for n in parts]
    positions = resample(raw["part_pos_m"][:, columns], raw["time_s"], times)
    quats = np.stack([Slerp(raw["time_s"], Rotation.from_quat(raw["part_quat_wxyz"][:, j], scalar_first=True))(times).as_quat(scalar_first=True) for j in columns], axis=1)
    reference = dict(time_s=times, object_state=body_reference(positions, quats, times))
    joint_keys = task["object"].get("reference_joint_keys", {})
    reference["object_q"] = np.stack([np.interp(times, raw["time_s"], raw[joint_keys[n]]) for n in task["object"]["joints"]], axis=-1) if joint_keys else np.empty((len(times), 0))
    reference["object_dq"] = velocity(reference["object_q"], times)
    surfaces = {p: sample_mesh(model, task["object"]["surface_geoms"][p], model.body(task["object"]["parts"][p]).id) for p in scoring}
    overrides = dict(item.split("=", 1) for item in args.object)
    if set(overrides)-set(robot["sides"]) or set(overrides.values())-set(scoring):
        raise ValueError("Invalid primary-part override")
    canonical = np.zeros((len(times), len(robot["joint_names"])), np.float32)
    guess = np.broadcast_to(np.clip(np.pi/36, robot["lower"], robot["upper"]), canonical.shape).copy()
    for side in robot["sides"]:
        points = resample(raw[f"{side}_joints_m"], raw["time_s"], times)
        targets = reference_from_points(model, robot, side, points, times)
        wrist_q = targets.pop("wrist_q")
        reference.update(targets)
        hand = robot["hands"][side]
        canonical[:, hand["indices"][:6]] = wrist_q
        distances = []
        for part, vertices in surfaces.items():
            index = parts.index(part)
            world = vertices[None]@Rotation.from_quat(quats[:, index], scalar_first=True).as_matrix().swapaxes(-1, -2)+positions[:, index, None]
            distances.append(np.stack([cKDTree(v).query(h[[4, 8, 12, 16, 20]])[0] for v, h in zip(world, reference[f"{side}_human"])]))
        distances = np.stack(distances, -1)
        part = overrides.get(side) or min(scoring, key=lambda p: (float(np.median(distances[..., scoring.index(p)].mean(axis=1))), p))
        body = model.body(task["object"]["parts"][part])
        hand.update(object_part=part, object_mass=float(body.mass[0]), object_com=body.ipos.tolist())
        reference.update({f"{side}_tip_distance_by_part": distances, f"{side}_tip_distance": distances.min(-1),
                          f"{side}_vertices": surfaces[part], f"{side}_bps": encode_bps(surfaces[part])})
        human = reference[f"{side}_human"]
        direction = (human[:, 0]+human[:, 9])/2-positions[:, parts.index(part)]
        length = np.linalg.norm(direction, axis=1, keepdims=True)
        rotation = Rotation.from_quat(reference[f"{side}_wrist"][:, 3:7], scalar_first=True).as_matrix()
        direction = np.where(length > 1e-10, direction/np.maximum(length, 1e-10), rotation[:, :, 2])
        wrist = WristReference(model, robot, side)
        # The outward IK initialization may lie outside workspace; project the guess only.
        position = human[:, 0]+.2*direction
        xyz = np.linalg.solve(wrist.translation_axes, (position-wrist.origin).T).T
        initial_wrist = wrist_q.copy()
        initial_wrist[:, :3] = xyz
        guess[:, hand["indices"][:6]] = np.clip(initial_wrist, wrist.lower, wrist.upper)
    reference["robot_q"] = canonical
    reference["robot_dq"] = velocity(canonical, times)
    optimized, ik = rsi.fit(source, task, robot, reference, guess, args.iterations, args.device)
    qpos, qvel = rsi.states(model, task, robot, reference, optimized)
    valid, reasons = rsi.validate(model, task, robot, reference, qpos, surfaces)
    np.savez_compressed(output / "reference.npz", **reference)
    np.savez_compressed(output / "rsi.npz", qpos=qpos, qvel=qvel, valid_start=valid, ticks=ticks)
    write_json(output / "rsi.json", dict(valid_start_count=int(valid.sum()), total_count=len(valid),
        source_ticks=ticks.tolist(), reasons=reasons, ik=ik, ik_iterations=args.iterations,
        wrist_reference_difference=(optimized[:, [i for s in robot["sides"] for i in robot["hands"][s]["indices"][:6]]]-canonical[:, [i for s in robot["sides"] for i in robot["hands"][s]["indices"][:6]]]).max(axis=0).tolist()))
    for p, vertices in surfaces.items():
        np.save(output / f"surface-{parts.index(p)}.npy", vertices)
    from .physics import check_initial_collision
    collisions = check_initial_collision(model, qpos[0], task["initialization"]["max_penetration_m"])
    write_json(output / "initialization.json", dict(valid=not collisions, collisions=collisions,
        robot_q=optimized[0].tolist(), joint_names=robot["joint_names"], velocity="zero", warmup_ticks=0,
        near_contact_diagnostic=reasons[0]))
    write_json(output / "robot_mapping.json", robot)
    compatibility = {s: signature(model, robot, s) for s in robot["sides"]}
    metadata = seal(output, "prepared.json", dict(format="maniptrans-h2r-prepared-v2", task_id=task["task_id"],
        bundle_id=task["bundle_id"], robot_manifest_sha256=task["robot_manifest_sha256"], part_names=parts,
        scoring_parts=scoring, hand_order=robot["sides"], object_root_names=list(task["object"].get("roots", {})),
        object_joint_names=list(task["object"]["joints"]), compatibility=compatibility,
        control_dt_s=1/60, physics_dt_s=1/120, decimation=2, physics_schedule="native-residual-v1", preparation_revision=3, ik=ik))
    print(f"Prepared {metadata['id']}; valid RSI starts {valid.sum()}/{len(valid)}; initial collision {'failed' if collisions else 'passed'}")
    return output
