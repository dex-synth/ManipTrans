"""Fit ManipTrans keypoint references on a sealed task's virtual-wrist robot."""

import argparse
from pathlib import Path
import shutil

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from .io import check_inputs, load_arrays, load_task, read_json, seal, write_json
from .robot import FIT_WEIGHTS, MANO_ORDER, initial_qpos, mapping


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


def fit(task_path, task, robot, arrays, iterations, device):
    import mujoco
    import pytorch_kinematics as pk
    import torch

    model = mujoco.MjModel.from_xml_path(str(task_path / task["scene"]))
    manifest_path = task_path / "scene/robot/assets/manifest.yaml"
    import yaml
    import xml.etree.ElementTree as ET
    manifest = yaml.safe_load(manifest_path.read_text())
    urdf = manifest_path.parent / manifest["entrypoints"][task["robot_name"]]["urdf"]
    tree = ET.fromstring(urdf.read_bytes())
    # Kinematics parser requires effort/velocity; these are never used for physics.
    for limit in tree.iter("limit"):
        limit.attrib.setdefault("effort", "0")
        limit.attrib.setdefault("velocity", "0")
    chain = pk.build_chain_from_urdf(ET.tostring(tree)).to(device=device, dtype=torch.float32)
    parameter_names = chain.get_joint_parameter_names()
    names = robot["joint_names"]
    if set(parameter_names) != set(names):
        raise ValueError("URDF joint set differs from the sealed robot")
    chain_order = [names.index(n) for n in parameter_names]
    data = mujoco.MjData(model)
    data.qpos[:] = initial_qpos(model, task)
    data.qpos[robot["qpos_indices"]] = 0
    mujoco.mj_forward(model, data)
    tracked = [b for side in robot["sides"] for b in robot["hands"][side]["bodies"]]
    # Verify the kinematic helper against task physics before using its gradients.
    probe = np.random.default_rng(0).uniform(-0.05, 0.05, len(names))
    data.qpos[robot["qpos_indices"]] = probe
    mujoco.mj_forward(model, data)
    transforms = chain.forward_kinematics(torch.tensor(probe[chain_order][None], dtype=torch.float32, device=device))
    for body in tracked:
        if not np.allclose(transforms[body].get_matrix()[0, :3, 3].detach().cpu(), data.body(body).xpos, atol=2e-5):
            raise ValueError(f"URDF/task FK differs at {body}")
    data.qpos[robot["qpos_indices"]] = 0
    mujoco.mj_forward(model, data)
    count = len(arrays["time_s"])
    guess = np.zeros((count, len(names)), dtype=np.float32)
    targets = []
    wrist_indices = []
    for side in robot["sides"]:
        hand = robot["hands"][side]
        human = arrays[f"{side}_joints_m"][:, MANO_ORDER]
        local = np.stack([data.body(b).xpos.copy() for b in hand["bodies"]])
        palm = [5, 9, 13, 17]
        source = local[palm] - local[:1]
        desired = human[:, palm] - human[:, :1]
        u, _, vt = np.linalg.svd(np.einsum("ni,tnj->tij", source, desired))
        correction = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
        correction[:, 2, 2] = np.linalg.det(vt.swapaxes(-1, -2) @ u.swapaxes(-1, -2))
        rotation = vt.swapaxes(-1, -2) @ correction @ u.swapaxes(-1, -2)
        idx = hand["indices"]
        guess[:, idx[:3]] = human[:, 0] - local[0]
        guess[:, idx[3:6]] = np.unwrap(Rotation.from_matrix(rotation).as_euler("XYZ"), axis=0)
        targets.append(human)
        wrist_indices.extend(idx[:6])
    target = torch.tensor(np.concatenate(targets, axis=1), dtype=torch.float32, device=device)
    weights = torch.tensor(FIT_WEIGHTS * len(robot["sides"]), dtype=torch.float32, device=device)
    q = torch.nn.Parameter(torch.tensor(guess, device=device))
    lower = torch.tensor(robot["lower"], device=device)
    upper = torch.tensor(robot["upper"], device=device)
    optimizer = torch.optim.Adam([q], lr=0.0004)
    for step in range(iterations):
        transforms = chain.forward_kinematics(q[:, chain_order].clamp(lower[chain_order], upper[chain_order]))
        points = torch.stack([transforms[b].get_matrix()[:, :3, 3] for b in tracked], dim=1)
        loss = ((points - target).norm(dim=-1) * weights).mean()
        optimizer.zero_grad()
        loss.backward()
        q.grad[:, wrist_indices] *= 2
        optimizer.step()
        with torch.no_grad():
            q.clamp_(lower, upper)
        if step % 100 == 0 or step == iterations - 1:
            print(f"IK {step + 1}/{iterations}: weighted distance {loss.item():.6f} m", flush=True)
    result = q.detach().cpu().numpy()
    # Store actual FK of the method reference; human targets remain separate.
    transforms = chain.forward_kinematics(q[:, chain_order])
    wrist_quat = []
    for side in robot["sides"]:
        matrix = transforms[robot["hands"][side]["bodies"][0]].get_matrix().detach().cpu().numpy()
        wrist_quat.append(Rotation.from_matrix(matrix[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]])
    return result, np.stack(wrist_quat, axis=1)


def sample_mesh(model, geom_names, body_id, count=1000):
    import mujoco
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


def prepare(args):
    import mujoco

    source, task = load_task(args.task)
    if args.bundle or args.robot_bundle:
        if not args.bundle or not args.robot_bundle:
            raise ValueError("Pass both --bundle and --robot-bundle, or use the self-contained --task")
        check_inputs(args.bundle, args.robot_bundle, source, task)
    output = Path(args.output).resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Output must be outside the input task")
    output.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copytree(source, output / "task")
        model = mujoco.MjModel.from_xml_path(str(source / task["scene"]))
        robot = mapping(model, task)
        reference = read_json(source / "reference.json")
        arrays = load_arrays(source / "reference.npz")
        times = arrays["time_s"]
        if not np.isclose(times[-1], task["end_tick"] * task["dt_s"], atol=1e-10) or np.any(np.diff(times) <= 0):
            raise ValueError("Reference clock differs from task")
        q, wrist_quat = fit(source, task, robot, arrays, args.iterations, args.device)
        part_order = reference["object_geometry"]["part_order"]
        part_names = list(task["object"]["parts"])
        columns = [part_order.index(n) for n in part_names]
        obj_state = body_reference(arrays["part_pos_m"][:, columns], arrays["part_quat_wxyz"][:, columns], times)
        targets = dict(item.split("=", 1) for item in args.object)
        unknown = set(targets) - set(robot["sides"])
        if unknown:
            raise ValueError(f"Unknown hand object bindings: {unknown}")
        prepared = {"time_s": times, "robot_q": q, "robot_dq": velocity(q, times), "object_state": obj_state}
        for i, side in enumerate(robot["sides"]):
            hand = robot["hands"][side]
            human = arrays[f"{side}_joints_m"][:, MANO_ORDER]
            if side not in targets:
                if len(part_names) != 1:
                    raise ValueError(f"Specify --object {side}=PART for a multi-part task")
                targets[side] = part_names[0]
            if targets[side] not in part_names:
                raise ValueError(f"Unknown object part: {targets[side]}")
            hand["object_part"] = targets[side]
            part_index = part_names.index(targets[side])
            body = model.body(task["object"]["parts"][targets[side]])
            hand["object_mass"] = float(body.mass[0])
            hand["object_com"] = body.ipos.tolist()
            vertices = sample_mesh(model, task["object"]["surface_geoms"][targets[side]], body.id)
            rot = Rotation.from_quat(obj_state[:, part_index, 3:7][:, [1, 2, 3, 0]]).as_matrix()
            world_vertices = vertices[None] @ rot.swapaxes(-1, -2) + obj_state[:, part_index, None, :3]
            from scipy.spatial import cKDTree
            distances = np.stack([cKDTree(v).query(h[[4, 8, 12, 16, 20]])[0] for v, h in zip(world_vertices, human)])
            prepared[f"{side}_human"] = human
            prepared[f"{side}_human_velocity"] = velocity(human, times)
            prepared[f"{side}_wrist"] = body_reference(human[:, 0], wrist_quat[:, i], times)
            prepared[f"{side}_tip_distance"] = distances
            prepared[f"{side}_vertices"] = vertices
            from .geometry import encode_bps
            prepared[f"{side}_bps"] = encode_bps(vertices)
        joint_keys = task["object"].get("reference_joint_keys", {"top_hinge": "object_hinge_rad"})
        if not set(task["object"].get("scoring_parts", part_names)) <= set(targets.values()):
            raise ValueError("Each scoring part must be assigned to a hand's object-tracking reward")
        prepared["object_q"] = np.stack([arrays[joint_keys[n]] for n in task["object"]["joints"]], axis=-1) if task["object"]["joints"] else np.empty((len(times), 0))
        prepared["object_dq"] = velocity(prepared["object_q"], times)
        np.savez_compressed(output / "reference.npz", **prepared)
        write_json(output / "robot_mapping.json", robot)
        from .physics import check_initial_collision
        initial = initial_qpos(model, task)
        initial[robot["qpos_indices"]] = q[0]
        collisions = check_initial_collision(model, initial, task["initialization"]["max_penetration_m"])
        write_json(output / "initialization.json", {"valid": not collisions, "collisions": collisions})
        metadata = seal(output, "prepared.json", {"format": "maniptrans-h2r-prepared-v1", "task_id": task["task_id"],
                        "bundle_id": task["bundle_id"], "robot_manifest_sha256": task["robot_manifest_sha256"],
                        "part_names": part_names, "ik_iterations": args.iterations, "control_dt_s": task["control_ticks"]*task["dt_s"]})
        print(f"Prepared {metadata['id']}; initial collision check: {'failed' if collisions else 'passed'}")
        return output
    except Exception:
        shutil.rmtree(output)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--bundle")
    parser.add_argument("--robot-bundle")
    parser.add_argument("--output", required=True)
    parser.add_argument("--object", action="append", default=[], metavar="SIDE=PART")
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument("--legacy-v1", action="store_true", help="Explicit legacy single-demonstration preparation")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.legacy_v1:
        prepare(args)
    else:
        from .prepare_v2 import prepare as prepare_v2
        prepare_v2(args)


if __name__ == "__main__":
    main()
