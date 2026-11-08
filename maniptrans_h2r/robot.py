"""Explicit Sharpa correspondence; physics and joint order come from the task."""

import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
MANO_ORDER = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
FIT_WEIGHTS = [1, 1, 1, 1, 25, 1, 1, 1, 20, 1, 1, 1, 10, 1, 1, 1, 7, 1, 1, 1, 5]


def mapping(model, task):
    import mujoco

    if task["robot_name"] not in {"virtual_wrist_sharpa_wave_right", "virtual_wrist_sharpa_wave_bimanual"}:
        raise ValueError("Currently implemented robot: common Sharpa Wave (right or bimanual)")
    semantics = task["semantics"]
    for joint in task["object"]["joints"].values():
        if int(model.joint(joint).type[0]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("Only scalar passive hinge object joints are currently supported")
    sides = [s for s in ("right", "left") if s in semantics["hands"]]
    if set(sides) != set(task.get("hand_sides", sides)):
        raise ValueError("Task hands differ from robot hands")
    names = semantics["joint_names"]
    result = {"joint_names": names, "sides": sides, "hands": {}}
    for side in sides:
        hand = semantics["hands"][side]
        wrist = [f"{side}_{a}_joint" for a in ("x", "y", "z", "roll", "pitch", "yaw")]
        tracked = [hand["wrist_body"]]
        for finger in FINGERS:
            links = ("MC", "PP", "DP", "fingertip") if finger == "thumb" else ("PP", "MP", "DP", "fingertip")
            tracked.extend(f"{side}_{finger}_{link}" for link in links)
        contact = [f"{side}_{finger}_DP" for finger in FINGERS]
        for body in tracked + contact:
            model.body(body)  # Fail before training on missing correspondence.
        ordered = wrist + hand["finger_joints"]
        indices = [names.index(n) for n in ordered]
        result["hands"][side] = dict(joints=ordered, indices=indices, bodies=tracked,
                                     contact_bodies=contact, mano_order=MANO_ORDER)
    if sorted(i for h in result["hands"].values() for i in h["indices"]) != list(range(len(names))):
        raise ValueError("Hand mappings must cover the entire common robot")
    joints = [model.joint(n) for n in names]
    result["qpos_indices"] = [int(j.qposadr[0]) for j in joints]
    result["qvel_indices"] = [int(j.dofadr[0]) for j in joints]
    result["lower"] = [float(j.range[0]) for j in joints]
    result["upper"] = [float(j.range[1]) for j in joints]
    result["actuator_indices"] = [int(model.actuator(n).id) for n in semantics["actuator_names"]]
    result["ctrl_joint_indices"] = [names.index(n) for n in semantics["actuator_joint_names"]]
    return result


def object_roots(task):
    obj = task["object"]
    if "roots" in obj:
        return obj["roots"]
    return {"object": {"body": obj["root_body"], "joint": obj.get("root_joint")}}


def initial_qpos(model, task):
    qpos = model.qpos0.copy()
    initial = task["initialization"]
    poses = initial.get("object_roots", {"object": initial.get("object_pose_wxyz")})
    for name, root in object_roots(task).items():
        if not root["joint"]:
            continue
        j = model.joint(root["joint"])
        qpos[int(j.qposadr[0]):int(j.qposadr[0]) + 7] = poses[name]
    for name, joint in task["object"]["joints"].items():
        qpos[int(model.joint(joint).qposadr[0])] = initial["object_joints"][name]
    return qpos


def weight_indices():
    result = {f"{finger}_tip": [4 * (i + 1)] for i, finger in enumerate(FINGERS)}
    result["level_1_joints"] = [1, 5, 9, 13, 17]
    result["level_2_joints"] = [i for i in range(1, 21) if i % 4 in (2, 3)]
    return result


def command_limits(model, robot):
    """Finite JSON bounds: unbounded wrists use the largest representable float32."""
    bound = float(np.finfo(np.float32).max)
    lower, upper = np.full(len(robot["joint_names"]), -bound), np.full(len(robot["joint_names"]), bound)
    for hand in robot["hands"].values():
        ids = hand["indices"][6:]
        lower[ids], upper[ids] = np.asarray(robot["lower"])[ids], np.asarray(robot["upper"])[ids]
    for actuator, joint in zip(robot["actuator_indices"], robot["ctrl_joint_indices"]):
        if model.actuator_ctrllimited[actuator]:
            lower[joint] = max(lower[joint], model.actuator_ctrlrange[actuator, 0])
            upper[joint] = min(upper[joint], model.actuator_ctrlrange[actuator, 1])
    if np.any(lower > upper):
        raise ValueError("Empty joint/actuator command range")
    robot.update(command_lower=lower.tolist(), command_upper=upper.tolist())
    return robot
