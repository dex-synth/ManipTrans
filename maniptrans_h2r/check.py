"""Local GPU checks of a deployment's private training world: clock, masked reset, terminal step, power and nominal physics."""

import argparse
from pathlib import Path

import torch

from .clock import CONTROL_DT_S, DECIMATION, PHYSICS_DT_S
from .evaluate import Controller
from .io import write_json
from .physics import gpu_audit
from .residual_env import ResidualEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = Path(args.deployment)
    controller = Controller(root, args.device)
    env = ResidualEnv(root / "prepared", 2, args.device, curriculum=False)
    robot = env.robot

    def initial():
        env.reset()
        assert not env.completed_controls.any()
        env.views["qpos"][:] = torch.as_tensor(env.rsi["qpos"][0], device=args.device)
        for key in ("qvel", "time", "qacc_warmstart"):
            env.views[key].zero_()
        for clock in (env.tick, env.elapsed_ticks, env.completed_controls):
            clock.zero_()
        env.views["ctrl"][:, robot["actuator_indices"]] = controller.initial[:, robot["ctrl_joint_indices"]]
        env.sim.forward()
        env.tip_force = env.contacts()
        env.control.reset([0, 1], controller.initial.expand(2, -1))

    def check_power():
        expected = env.views["qfrc_actuator"][:, robot["qvel_indices"]]*env.views["qvel"][:, robot["qvel_indices"]]
        torch.testing.assert_close(env.last_power, expected, rtol=0, atol=0)

    initial()
    actions = torch.zeros((2, 2*env.action_dim), device=args.device)
    env.step(actions)
    check_power()
    # A reset of one world must leave every other world untouched.
    kept = {k: v[0].clone() for k, v in env.views.items()}
    history = {s: v[0].clone() for s, v in env.contact_history.items()}
    forces = {s: v[0].clone() for s, v in env.tip_force.items()}
    previous = env.control.previous[0].clone()
    env.done[:] = False
    env.done[1] = True
    env.reset_done()
    assert env.completed_controls.tolist() == [1, 0]
    for key, value in kept.items():
        torch.testing.assert_close(env.views[key][0], value, rtol=0, atol=0)
    for side in env.sides:
        torch.testing.assert_close(env.contact_history[side][0], history[side], rtol=0, atol=0)
        torch.testing.assert_close(env.tip_force[side][0], forces[side], rtol=0, atol=0)
    torch.testing.assert_close(env.control.previous[0], previous, rtol=0, atol=0)
    # One second of task time is 1/CONTROL_DT_S controls of DECIMATION training physics steps each.
    initial()
    controls = round(1/CONTROL_DT_S)
    for _ in range(controls):
        env.done.zero_()  # Clock diagnostic only: allow native tracking failures.
        env.step(torch.zeros_like(actions))
    assert env.tick.tolist() == [controls*DECIMATION]*2 and env.completed_controls.tolist() == [controls]*2
    torch.testing.assert_close(env.views["time"], torch.ones_like(env.views["time"]), rtol=0, atol=2e-6)
    # A control that reaches the task end takes only the remaining physics step.
    env.done.zero_()
    env.tick[:] = env.end_tick-1
    before = env.views["time"].clone()
    _, _, done, info = env.step(torch.zeros_like(actions))
    check_power()
    assert done.all() and env.tick.tolist() == [env.end_tick]*2 and not info["time_outs"].any()
    torch.testing.assert_close(env.views["time"]-before, torch.full_like(before, PHYSICS_DT_S), rtol=0, atol=1e-7)
    audit = gpu_audit(env.cpu, env.model, env.task.task_id, env.metadata["scene_sha256"])
    assert audit["fields"] == env.audit["fields"] and audit["options"] == env.audit["options"]
    write_json(args.output, dict(task_id=env.task.task_id, control_dt_s=CONTROL_DT_S, physics_dt_s=PHYSICS_DT_S,
               masked_reset=True, one_second_clock=True, shortened_terminal_step=True, post_step_power=True,
               nominal_physics_unchanged=True))
    print(f"Passed mjlab GPU contracts: {args.output}")


if __name__ == "__main__":
    main()
