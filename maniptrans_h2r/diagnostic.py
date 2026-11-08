"""Fixed training-corpus tracking diagnostics; never a manipulation benchmark."""

from pathlib import Path

import numpy as np
import torch

from .io import load_artifact, read_json, write_json
from .policy import Policy, build
from .residual_reward import hand_inputs
from .robot import weight_indices


@torch.no_grad()
def tracking(model, prepared, device, allow_partial, *, stochastic=False):
    from .imitator_env import ImitatorEnv
    torch.manual_seed(43)
    manifest = read_json(prepared / "diagnostic_manifest.json")
    selected = manifest["windows"]
    if not selected:
        return dict(label=manifest["overlap"], windows=[], count=0)
    count = min(32, len(selected))
    env = ImitatorEnv(prepared, count, device, seed=43, allow_partial=allow_partial)
    env.fixed_scale = .7
    env.diagnostic = not stochastic
    model.eval()
    results = []
    for start in range(0, len(selected), count):
        group = selected[start:start+count]
        env.fixed_windows = group + [group[-1]]*(count-len(group))
        obs = env.reset()
        active = torch.arange(count, device=device) < len(group)
        sums = {}
        start_positions = env.motion.positions.copy()
        samples = torch.zeros(count, device=device)
        while active.any():
            result = model(dict(is_train=False, obs=dict(obs)))
            raw = result["actions"] if stochastic else result["mus"]
            actions = -1+2*torch.sigmoid(2*raw)
            obs, reward, done, info = env.step(actions)
            state, target = env.state(), env.target(0)
            hand = env.robot["hands"][env.side]
            bodies = torch.stack([state["bodies"][b] for b in hand["bodies"]], 1)
            wrist = target[f"{env.side}_wrist"]
            points = (bodies[:, 1:, :3]-target[f"{env.side}_human"][:, 1:]).norm(dim=-1)
            _, goal = hand_inputs(env.robot, env.side, state, target, env.last_power)
            errors = dict(wrist_position_m=(bodies[:, 0, :3]-wrist[:, :3]).norm(dim=-1),
                wrist_rotation_rad=2*torch.acos((bodies[:, 0, 3:7]*wrist[:, 3:7]).sum(-1).abs().clamp(max=1)),
                wrist_velocity_m_s=(bodies[:, 0, 7:10]-wrist[:, 7:10]).norm(dim=-1),
                wrist_angular_velocity_rad_s=(bodies[:, 0, 10:13]-wrist[:, 10:13]).norm(dim=-1),
                mean_keypoint_m=points.mean(-1), mean_reward=reward,
                finger_power_w=goal["power"], wrist_power_w=goal["wrist_power"])
            groups = {name: points[:, [i-1 for i in ids]].mean(-1) for name, ids in weight_indices().items()}
            errors.update({name+"_m": value for name, value in groups.items()})
            for key, values in errors.items():
                sums.setdefault(key, torch.zeros(count, device=device))[active] += values[active]
            samples[active] += 1
            for row in (active & done).nonzero().flatten().tolist():
                window = env.motion.windows[group[row]]
                thresholds = dict(thumb_tip=.04, index_tip=.045, middle_tip=.05, ring_tip=.06,
                                  pinky_tip=.06, level_1_joints=.07, level_2_joints=.08)
                reasons = [k for k, threshold in thresholds.items()
                           if env.completed_controls[row] > 20 and groups[k][row] > threshold]
                results.append(dict(window=group[row], dataset=window["dataset"], source_side=window["source_side"],
                    success=bool(env.success_buf[row]), transitions=int(samples[row]),
                    start_sample=int(start_positions[row]),
                    coverage=float(samples[row]/(window["length"]-1)),
                    remaining_coverage=float(samples[row]/(window["length"]-1-start_positions[row])),
                    failure_reasons=(reasons or ["native_velocity_sanity"]) if env.failure_buf[row] else [],
                    **{k: float(v[row]/samples[row]) for k, v in sums.items()}))
            active &= ~done
            if active.any():
                obs = env.reset_done()[0]
    return dict(label=manifest["overlap"], count=len(results), stochastic=stochastic, seed=43,
                strict_scale=.7, windows=results)


def diagnostic_callback(prepared, output, device, allow_partial, final_epoch):
    destination = output / "diagnostics"
    destination.mkdir()
    def callback(model, epoch):
        write_json(destination / f"epoch-{epoch}.json", tracking(model, prepared, device, allow_partial))
        if epoch == final_epoch:
            write_json(destination / "final-rsi-sampled.json", tracking(model, prepared, device, allow_partial, stochastic=True))
    return callback


@torch.no_grad()
def residual_tracking(model, prepared, device):
    """Start at the evaluation pose; report the prefix before strict failure.

    This uses a separate nominal training world, not the benchmark evaluator.
    """
    from .residual_env import ResidualEnv
    initialization = read_json(prepared / "initialization.json")
    report = dict(label="full-task tracking diagnostic", seed=43, strict_scale=.7,
                  start_tick=0, initial_velocity="zero", physics_schedule="nominal")
    if not initialization["valid"]:
        return dict(report, status="initialization_failed", success=False, coverage=0.,
                    collisions=initialization["collisions"])
    env = ResidualEnv(prepared, 1, device, seed=43, from_start=True, curriculum=False)
    env.fixed_scale = .7
    policy = Policy(model, env.action_dim, {"diagnostic": 43}, device)
    obs = env.observe()
    sums, transitions = {}, 0
    while True:
        obs, reward, done, _ = env.step(policy(obs, ["diagnostic"]))
        state, target = env.state(), env.target(0)
        errors = {"reward": float(reward[0])}
        for side in env.sides:
            hand = env.robot["hands"][side]
            points = torch.stack([state["bodies"][b][:, :3] for b in hand["bodies"]], 1)
            errors[f"{side}/keypoint_error_m"] = float((points-target[f"{side}_human"]).norm(dim=-1).mean())
        for part in env.part_indices:
            actual, expected = state["objects"][:, part], target["object_state"][:, part]
            name = env.metadata["part_names"][part]
            errors[f"{name}/position_error_m"] = float((actual[:, :3]-expected[:, :3]).norm())
            errors[f"{name}/rotation_error_rad"] = float(2*torch.acos((actual[:, 3:7]*expected[:, 3:7]).sum().abs().clamp(max=1)))
        for key, value in errors.items():
            sums[key] = sums.get(key, 0.)+value
        transitions += 1
        if done[0]:
            break
    return dict(report, status="completed", task_id=env.task["task_id"], success=bool(env.success_buf[0]),
                failure=bool(env.failure_buf[0]), terminal_tick=int(env.tick[0]), end_tick=env.task["end_tick"],
                coverage=float(env.tick[0]/env.task["end_tick"]), transitions=transitions,
                mean={k: v/transitions for k, v in sums.items()})


def residual_diagnostic_callback(prepared, output, device):
    destination = output / "diagnostics"
    destination.mkdir()
    def callback(model, epoch):
        write_json(destination / f"epoch-{epoch}.json", residual_tracking(model, prepared, device))
    return callback


def check_imitator(args):
    root, meta = load_artifact(args.imitator, "imitator.json")
    prepared, input_meta = load_artifact(args.prepared)
    if input_meta["id"] != meta["prepared_id"]:
        raise ValueError("Diagnostic corpus/reference differs from the trained imitator")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    model = build(meta["params"], meta["dimensions"], meta["action_dim"], args.device)
    model.load_state_dict(torch.load(root / meta["checkpoint"], map_location=args.device, weights_only=True)["model"])
    for stochastic in (False, True):
        write_json(output / ("rsi-sampled.json" if stochastic else "mean-from-start.json"),
                   tracking(model, prepared, args.device, meta["smoke"], stochastic=stochastic))
