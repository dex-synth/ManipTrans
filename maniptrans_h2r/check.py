"""Local GPU contract checks against a prepared task and optional bench rollout."""

import argparse
from pathlib import Path

import numpy as np
import torch
import warp as wp

from .evaluate import Controller
from .io import read_json, write_json
from .mjwarp_env import MJWarpEnv
from .observations import protocol_state
from .policy import Policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--run", help="Optional nominal bench physical rollout to replay for one control interval")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    deployment = Path(args.deployment)
    if read_json(deployment / "deployment.json")["format"] == "maniptrans-h2r-deployment-v2":
        from .check_v2 import check
        check(args)
        return
    client = Controller(deployment, args.device)
    env = MJWarpEnv(deployment / "prepared", "residual", client.robot["sides"], 2, args.device, client.manifest["control"])
    task, robot = env.task, env.robot
    hello = dict(type="hello", protocol="h2r-evaluation-v1", **{k: task[k] for k in
                 ("task_id", "bundle_id", "robot_manifest_sha256", "robot_name", "dt_s", "control_ticks", "end_tick", "object", "observation")},
                 joint_names=robot["joint_names"], joint_units=task["semantics"]["joint_units"],
                 episodes=[dict(episode_id=f"e{i}", controller_seed=i+19) for i in range(2)])
    client.start(hello)
    policy = Policy(client.model, client.manifest["action_dim"], {f"e{i}": i+19 for i in range(2)}, args.device)
    state = env.state()
    frames = []
    for i in range(2):
        records = []
        for side, hand in robot["hands"].items():
            for finger, body in enumerate(hand["contact_bodies"]):
                records.append(dict(body1=body, body2=task["object"]["parts"][hand["object_part"]],
                                    force_on_body2_world=(-state["tip_force"][side][i, finger]).tolist()))
        frames.append(dict(episode_id=f"e{i}", tick=0, done=False,
            **{k: state[k][i].tolist() for k in ("robot_q", "robot_dq", "object_q", "object_dq")},
            bodies={k: v[i].tolist() for k, v in state["bodies"].items()}, contacts=dict(solve_tick=None, records=records)))
    reconstructed = protocol_state(frames, robot, task, args.device)
    obs = env.features(state, env.target(task["control_ticks"]), env.sides, env.stage)
    obs_wire = env.features(reconstructed, env.target(task["control_ticks"]), env.sides, env.stage)
    for key in obs:
        torch.testing.assert_close(obs[key], obs_wire[key], rtol=0, atol=1e-6)
    commands = client.step(dict(type="frames", done=False, episodes=frames))["commands"]
    actions = policy(obs, ["e0", "e1"])
    targets = env.control.decode(actions, env.target(task["control_ticks"])["robot_q"], env.sides, True)
    wire_targets = torch.tensor([[commands[f"e{i}"]["joint_targets"][n] for n in robot["joint_names"]]
                                 for i in range(2)], device=args.device)
    torch.testing.assert_close(targets, wire_targets, rtol=1e-5, atol=1e-6)
    env.reset()
    env.step(actions)
    kept = {k: env.views[k][0].clone() for k in ("qpos", "qvel", "time", "qacc_warmstart")}
    kept_control = env.control.previous[0].clone()
    kept_force = {s: f[0].clone() for s, f in env.tip_force.items()}
    env.done[:] = False
    env.done[1] = True
    env.reset_done()
    for key, value in kept.items():
        torch.testing.assert_close(env.views[key][0], value, rtol=0, atol=0)
    torch.testing.assert_close(env.control.previous[0], kept_control, rtol=0, atol=0)
    for side, value in kept_force.items():
        torch.testing.assert_close(env.tip_force[side][0], value, rtol=0, atol=0)
    # Exercise a shortened final interval on a diagnostic clock without changing task files.
    env.task = dict(env.task, end_tick=task["control_ticks"]+3)
    _, _, done, info = env.step(actions)
    assert env.tick.tolist() == [task["control_ticks"]+3, task["control_ticks"]]
    assert done[0] and not info["time_outs"].any()
    env.task = task
    env.reset()
    report = dict(task_id=task["task_id"], feature_and_target_parity=True, masked_reset=True,
                  terminal_observation=True, shortened_final_interval=True, task_physics_audit=env.audit)
    if args.run:
        run = Path(args.run)
        manifest = read_json(run / "run.json")
        if manifest["task_id"] != task["task_id"] or manifest["mode"] != "sim":
            raise ValueError("Replay requires the same task with nominal physics")
        seed = manifest["seeds"][0]
        import json
        rows = [json.loads(line) for line in (run / f"seed-{seed}/physical.jsonl").read_text().splitlines()]
        q_errors, force_errors = [], []
        # Bench records the submitted target after each interval; row 1 contains tick-zero control.
        qorder = robot["qpos_indices"]
        for row in rows[1:task["control_ticks"]+1]:
            target = torch.tensor(row["effective_target"], device=args.device)
            env.views["ctrl"][:, robot["actuator_indices"]] = target[robot["ctrl_joint_indices"]]
            wp.capture_launch(env.step_graph)
            q_errors.append(float(np.max(np.abs(env.views["qpos"][0, qorder].cpu().numpy()-row["robot_q"]))))
            # Bench actuator_force is in hello joint order, independently of XML actuator order.
            actual = wp.to_torch(env.data.actuator_force)[0, robot["actuator_indices"]].cpu().numpy()
            force_errors.append(float(np.max(np.abs(actual-np.asarray(row["actuator_force"])[robot["ctrl_joint_indices"]]))))
        report["bench_first_interval"] = dict(max_q_error=max(q_errors), max_force_error=max(force_errors),
            backend=manifest["backend"], ticks=len(q_errors),
            note="MJWarp evaluator may retain its default float32 solver tolerance floor; training uses sealed tolerance.")
        if max(q_errors) > 2e-5 or max(force_errors) > .001:
            raise AssertionError(f"Training/evaluation physics disagree: {report['bench_first_interval']}")
    write_json(args.output, report)
    print(f"Passed GPU contracts: {args.output}")


if __name__ == "__main__":
    main()
