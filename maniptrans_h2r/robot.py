"""Explicit Sharpa correspondence; physics and joint order come from the task."""

import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
MANO_ORDER = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
FIT_WEIGHTS = [1, 1, 1, 1, 25, 1, 1, 1, 20, 1, 1, 1, 10, 1, 1, 1, 7, 1, 1, 1, 5]
SIDES = ("right", "left")  # Feature and action order of a bimanual policy.


def tracked_links(robot_name):
    """Per finger, the four links matched to MANO keypoints; this table is the method's own."""
    if robot_name != "sharpa_wave":
        raise ValueError(f"No ManipTrans keypoint correspondence for robot {robot_name!r}; implemented: sharpa_wave")
    return {finger: ("MC", "PP", "DP", "fingertip") if finger == "thumb" else ("PP", "MP", "DP", "fingertip")
            for finger in FINGERS}


def mapping(model, task):
    """Joint, body and actuator indices of the task's hands in the task model."""
    import mujoco

    links, semantics = tracked_links(task.robot.name), task.robot.semantics
    for joint in task.object_binding["joints"].values():
        if int(model.joint(joint).type[0]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("Only scalar passive hinge object joints are currently supported")
    sides = [side for side in SIDES if side in task.hand_sides]
    names = semantics["joint_names"]
    result = {"joint_names": names, "sides": sides, "hands": {}}
    for side in sides:
        hand = semantics["hands"][side]
        if len(hand["wrist_translation_joints"]) != 3 or len(hand["wrist_rotation_joints"]) != 3:
            raise ValueError("The wrist action needs three translation and three rotation joints")
        tracked = [hand["wrist_body"]] + [f"{side}_{finger}_{link}" for finger in FINGERS for link in links[finger]]
        contact = [f"{side}_{finger}_DP" for finger in FINGERS]
        for body in tracked + contact:
            model.body(body)  # Fail before training on missing correspondence.
        ordered = hand["wrist_translation_joints"] + hand["wrist_rotation_joints"] + hand["finger_joints"]
        result["hands"][side] = dict(joints=ordered, indices=[names.index(n) for n in ordered], bodies=tracked,
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


def initial_qpos(model, task):
    """The task's start state: objects at reference frame 0, the robot at the model default."""
    qpos = model.qpos0.copy()
    binding, state = task.object_binding, task.initial_object_state
    for name, root in binding["roots"].items():
        address = int(model.joint(root["joint"]).qposadr[0])
        qpos[address:address+7] = state["object_roots"][name]
    for name, joint in binding["joints"].items():
        qpos[int(model.joint(joint).qposadr[0])] = state["object_joints"][name]
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
