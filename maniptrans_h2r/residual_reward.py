"""Native reward calls with explicit power, episode clock and all-part aggregation."""

import torch

from lib.rl.rewards import imitator_reward, manipulation_reward
from .observations import xyzw
from .robot import weight_indices


def hand_inputs(robot, side, state, target, power):
    hand = robot["hands"][side]
    ids = hand["indices"]
    bodies = torch.stack([state["bodies"][b] for b in hand["bodies"]], 1).clone()
    bodies[..., 3:7] = xyzw(bodies[..., 3:7])
    wrist = target[f"{side}_wrist"]
    actual = dict(base_state=bodies[:, 0], joints_state=bodies,
                  q=state["robot_q"][:, ids[6:]], dq=state["robot_dq"][:, ids[6:]])
    goal = dict(wrist_pos=wrist[:, :3], wrist_quat=xyzw(wrist[:, 3:7]), wrist_vel=wrist[:, 7:10],
                wrist_ang_vel=wrist[:, 10:13], joints_pos=target[f"{side}_human"][:, 1:],
                joints_vel=target[f"{side}_human_velocity"][:, 1:],
                power=power[:, ids[6:]].abs().sum(-1),
                wrist_power=power[:, ids[:3]].sum(-1).abs()+power[:, ids[3:6]].sum(-1).abs())
    return actual, goal


def native_call(function, actual, goal, actions, elapsed_ticks, scale):
    progress = elapsed_ticks.float()/2
    return function(torch.zeros_like(elapsed_ticks), progress, progress, actions, actual, goal,
                    torch.full_like(progress, 1e9), scale, weight_indices())


def hand_reward(robot, side, state, target, power, actions, elapsed_ticks, scale):
    actual, goal = hand_inputs(robot, side, state, target, power)
    result = native_call(imitator_reward, actual, goal, actions, elapsed_ticks, scale)
    return result[0], result[3], result[4]


def residual_reward(robot, part_indices, state, target, power, actions, elapsed_ticks, scale, contact_history):
    rewards, failures, details = [], [], {}
    for side in robot["sides"]:
        actual, goal = hand_inputs(robot, side, state, target, power)
        hand = native_call(imitator_reward, actual, goal, actions, elapsed_ticks, scale)
        object_rewards, contact = [], None
        for part in part_indices:
            for values, source in ((actual, state["objects"][:, part]), (goal, target["object_state"][:, part])):
                values.update(manip_obj_pos=source[:, :3], manip_obj_quat=xyzw(source[:, 3:7]),
                              manip_obj_vel=source[:, 7:10], manip_obj_ang_vel=source[:, 10:13])
            goal.update(tip_force=state["tip_force"][side], tips_distance=target[f"{side}_tip_distance"],
                        tip_contact_state=contact_history[side])
            result = native_call(manipulation_reward, actual, goal, actions, elapsed_ticks, scale)
            terms = result[4]
            object_rewards.append(5*terms["reward_obj_pos"]+terms["reward_obj_rot"]+.1*terms["reward_obj_vel"]+.1*terms["reward_obj_ang_vel"])
            contact = terms["reward_finger_tip_force"]
            failures.append(result[3])
            details.update({f"{side}/part_{part}/{k}": v for k, v in terms.items() if "obj" in k})
        rewards.append(hand[0]+contact+torch.stack(object_rewards).mean(0))
        details.update({f"{side}/{k}": v for k, v in hand[4].items()})
        details[f"{side}/reward_finger_tip_force"] = contact
    return torch.stack(rewards).sum(0), torch.stack(failures).any(0), details
