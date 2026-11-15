"""Complete task targets and independently optimized reference-state resets, from one task directory."""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from . import rsi
from .clock import CONTROL_DT_S, DECIMATION, PHYSICS_DT_S, training_ticks
from .geometry import encode_bps
from .h2r_io import export_task, load_task
from .hand_reference import WristReference, reference_from_points
from .io import seal, write_json
from .physics import MAX_PENETRATION_M, check_initial_collision, verify_sealed
from .robot import command_limits, mapping
from .robot_physics import signature


def velocity(values, times):
    return gaussian_filter1d(np.gradient(values, times, axis=0), 2, axis=0, mode="nearest")


def body_reference(positions, quaternions, times):
    shape = positions.shape
    rotations = Rotation.from_quat(quaternions.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape(*shape[:-1], 3, 3)
    relative = rotations[1:] @ rotations[:-1].swapaxes(-1, -2)
    angular = Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec().reshape(len(times)-1, *shape[1:])
    angular /= np.diff(times).reshape(-1, *([1]*(angular.ndim-1)))
    angular = gaussian_filter1d(np.concatenate([angular, angular[-1:]]), 2, axis=0, mode="nearest")
    return np.concatenate([positions, quaternions, velocity(positions, times), angular], axis=-1)


def sample_mesh(model, geom_names, body_id, count=1000):
    import trimesh

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pieces = []
    for name in geom_names:
        geom = model.geom(name)
        mesh_id = int(geom.dataid[0])
        if int(geom.type[0]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            raise ValueError(f"Expected a sealed object visual mesh: {name}")
        v0, nv = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        f0, nf = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
        vertices = model.mesh_vert[v0:v0+nv].copy()
        world = vertices @ data.geom(name).xmat.reshape(3, 3).T + data.geom(name).xpos
        local = (world - data.xpos[body_id]) @ data.xmat[body_id].reshape(3, 3)
        pieces.append(trimesh.Trimesh(local, model.mesh_face[f0:f0+nf], process=False))
    return trimesh.sample.sample_surface(trimesh.util.concatenate(pieces), count, seed=0)[0]


def resample(values, source_times, times):
    shape = values.shape[1:]
    flat = values.reshape(len(values), -1)
    return np.stack([np.interp(times, source_times, flat[:, i]) for i in range(flat.shape[1])], -1).reshape(len(times), *shape)


def prepare(args):
    import torch
    torch.set_num_threads(4)
    task = load_task(args.task)
    demo, binding = task.demo, task.object_binding
    output = Path(args.output).resolve()
    if output.is_relative_to(task.path):
        raise ValueError("Output must be outside the task")
    output.mkdir(parents=True, exist_ok=False)
    export_task(task, output / "task")
    model = mujoco.MjModel.from_xml_path(str(task.scene_xml))
    verify_sealed(model, task.compiled)
    robot = command_limits(model, mapping(model, task))
    # The method's reference grid: one sample per control period, plus the task endpoint.
    end = training_ticks(task.duration_s)
    times = np.unique(np.r_[np.arange(0, end+1, DECIMATION), end])*PHYSICS_DT_S
    parts, scoring = list(binding["parts"]), list(binding["scoring_parts"])
    poses = [demo.part_pose(part) for part in parts]
    positions = resample(np.stack([p for p, _ in poses], 1), demo.time_s, times)
    quats = np.stack([Slerp(demo.time_s, Rotation.from_quat(q, scalar_first=True))(times).as_quat(scalar_first=True)
                      for _, q in poses], axis=1)
    reference = dict(time_s=times, object_state=body_reference(positions, quats, times))
    joints = [joint for obj in demo.objects.values() for joint in obj.joints]
    if [joint.name for joint in joints] != list(binding["joints"]):
        raise ValueError("Object joint order differs from the evaluation binding")
    reference["object_q"] = (np.stack([np.interp(times, demo.time_s, joint.position_rad) for joint in joints], axis=-1)
                             if joints else np.empty((len(times), 0)))
    reference["object_dq"] = velocity(reference["object_q"], times)
    surfaces = {p: sample_mesh(model, binding["surface_geoms"][p], model.body(binding["parts"][p]).id) for p in scoring}
    overrides = dict(item.split("=", 1) for item in args.object)
    if set(overrides)-set(robot["sides"]) or set(overrides.values())-set(scoring):
        raise ValueError("Invalid primary-part override")
    canonical = np.zeros((len(times), len(robot["joint_names"])), np.float32)
    guess = np.broadcast_to(np.clip(np.pi/36, robot["lower"], robot["upper"]), canonical.shape).copy()
    for side in robot["sides"]:
        points = resample(demo.hands[side].joints_m, demo.time_s, times)
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
        body = model.body(binding["parts"][part])
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
    optimized, ik = rsi.fit(task, robot, reference, guess, args.iterations, args.device)
    qpos, qvel = rsi.states(model, task, robot, reference, optimized)
    valid, reasons = rsi.validate(model, task, robot, reference, qpos, surfaces)
    np.savez_compressed(output / "reference.npz", **reference)
    np.savez_compressed(output / "rsi.npz", qpos=qpos, qvel=qvel, valid_start=valid, time_s=times)
    wrist_ids = [i for s in robot["sides"] for i in robot["hands"][s]["indices"][:6]]
    write_json(output / "rsi.json", dict(valid_start_count=int(valid.sum()), total_count=len(valid),
        reasons=reasons, ik=ik, ik_iterations=args.iterations, max_penetration_m=MAX_PENETRATION_M,
        wrist_reference_difference=(optimized[:, wrist_ids]-canonical[:, wrist_ids]).max(axis=0).tolist()))
    for p, vertices in surfaces.items():
        np.save(output / f"surface-{parts.index(p)}.npy", vertices)
    collisions = check_initial_collision(model, qpos[0])
    write_json(output / "initialization.json", dict(valid=not collisions, collisions=collisions,
        max_penetration_m=MAX_PENETRATION_M, robot_q=optimized[0].tolist(), joint_names=robot["joint_names"],
        near_contact_diagnostic=reasons[0]))
    write_json(output / "robot_mapping.json", robot)
    metadata = seal(output, "prepared", dict(task_id=task.task_id, demo_id=demo.demo_id, robot_id=task.robot.robot_id,
        dataset=task.dataset, profile=task.profile, part_names=parts, object_bodies=binding["parts"],
        scoring_parts=scoring, object_joint_names=list(binding["joints"]),
        gravity_m_s2=float(np.linalg.norm(model.opt.gravity)),
        compatibility={s: signature(model, robot, s) for s in robot["sides"]},
        control_dt_s=CONTROL_DT_S, physics_dt_s=PHYSICS_DT_S, physics_schedule="native-residual-v1", ik=ik))
    print(f"Prepared {metadata['id']}; valid RSI starts {valid.sum()}/{len(valid)}; initial collision {'failed' if collisions else 'passed'}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="Task directory (standalone or inside a suite)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--object", action="append", default=[], metavar="SIDE=PART")
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    prepare(args)


if __name__ == "__main__":
    main()
