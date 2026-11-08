"""Train virtual-wrist imitators then a frozen-base residual policy on MJWarp."""

import argparse
import gc
import math
from copy import deepcopy
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import torch
from rl_games.common.algo_observer import AlgoObserver
from rl_games.common.tr_helpers import DefaultRewardsShaper

from lib.rl.agent import PPOAgent
from .io import load_artifact, read_json, seal, write_json
from .mjwarp_env import MJWarpEnv
from .policy import parameters, register


def control_config(control_dt_s):
    # Native moving average 0.4 at 60 Hz, expressed as the same time constant.
    time_constant = -(1/60)/math.log(1-0.4)
    return dict(wrist_translation_m=0.05, wrist_rotation_rad=0.3,
                moving_average=1-math.exp(-control_dt_s/time_constant), smoothing_time_s=time_constant)


def train_stage(prepared, output, stage, sides, bases, args):
    control = control_config(read_json(prepared / "prepared.json")["control_dt_s"])
    env = MJWarpEnv(prepared, stage, sides, args.worlds, args.device, control)
    dimensions = {k: s.shape[0] for k, s in env.observation_space.spaces.items()}
    params = parameters(dimensions, env.features.base_dimensions, sides, bases, env.action_dim, args.small_network)
    saved = deepcopy(params)
    name = stage + "_" + "_".join(sides)
    config = dict(name=name, full_experiment_name=name, env_name="h2r_mjwarp", env_info=env.get_env_info(), vec_env=env,
                  device=args.device, num_actors=args.worlds, train_dir=str(output / "training"),
                  features={"observer": AlgoObserver()}, reward_shaper=DefaultRewardsShaper(scale_value=1),
                  use_pid_control=False, ppo=True, normalize_input=True, normalize_input_excluded_keys=[],
                  normalize_value=True, normalize_advantage=True, value_bootstrap=True,
                  gamma=0.99, tau=0.95, learning_rate=5e-4, lr_schedule="adaptive", schedule_type="standard",
                  kl_threshold=0.008, score_to_win=10000, max_epochs=args.epochs, early_stop_epochs=args.epochs+1,
                  save_best_after=10, save_frequency=0, print_stats=True, grad_norm=1.0, entropy_coef=0,
                  truncate_grads=True, e_clip=0.2, horizon_length=args.horizon,
                  minibatch_size=min(args.worlds*args.horizon, 1024), mini_epochs=args.mini_epochs,
                  critic_coef=4, clip_value=True, bounds_loss_coef=0.0001, use_soft_clamp=True,
                  store_base_actions=stage == "residual")
    params["config"] = config
    register()
    agent = PPOAgent("run", params)
    before = {k: v.detach().clone() for k, v in agent.model.named_parameters() if v.requires_grad}
    frozen = {k: v.detach().clone() for k, v in agent.model.named_parameters() if not v.requires_grad}
    agent.train()
    changed = any(not torch.equal(before[k], v) for k, v in agent.model.named_parameters() if k in before)
    if not changed or any(not torch.equal(frozen[k], v) for k, v in agent.model.named_parameters() if k in frozen):
        raise RuntimeError("Training did not update the policy, or modified a frozen base")
    checkpoint = output / f"{name}.pth"
    # Deployment uses tensor-only state; no optimizer or simulator state is loaded online.
    torch.save({"model": agent.model.state_dict()}, checkpoint)
    write_json(output / f"{name}_physics.json", env.audit)
    write_json(output / f"{name}_training.json", dict(epochs=agent.epoch_num, frames=agent.frame,
               changed=changed, frozen_parameters=len(frozen), seed=args.seed, worlds=args.worlds,
               horizon=args.horizon, mini_epochs=args.mini_epochs, small_network=args.small_network))
    agent.writer.close()
    return checkpoint, saved, dimensions, env.action_dim


def legacy_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--worlds", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--mini-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--small-network", action="store_true", help="Explicit smoke-test network variant")
    args = parser.parse_args()
    if min(args.worlds, args.epochs, args.horizon, args.mini_epochs) < 1 or args.worlds*args.horizon % min(args.worlds*args.horizon, 1024):
        parser.error("Positive sizes and a batch divisible by the minibatch size are required")
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    source, metadata = load_artifact(args.prepared)
    output = Path(args.output).resolve()
    if output.is_relative_to(source):
        raise ValueError("Output must be outside the prepared artifact")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source, output / "prepared")
    robot = read_json(source / "robot_mapping.json")
    bases = {}
    for side in robot["sides"]:
        path, _, _, _ = train_stage(output / "prepared", output, "imitator", [side], {}, args)
        bases[side] = str(path)
        gc.collect()
        torch.cuda.empty_cache()
    path, params, dimensions, action_dim = train_stage(output / "prepared", output, "residual", robot["sides"], bases, args)
    for side, prefix in (("right", "rh"), ("left", "lh")):
        if side in bases:
            params["model"][f"{prefix}_base_model_checkpoint"] = Path(bases[side]).name
            params["network"]["base_model"][f"{prefix}_checkpoint"] = Path(bases[side]).name
    seal(output, "deployment.json", dict(format="maniptrans-h2r-deployment-v1", method="maniptrans-mjwarp",
         controller="virtual-wrist-position-v1", prepared_id=metadata["id"], task_id=metadata["task_id"],
         bundle_id=metadata["bundle_id"], robot_manifest_sha256=metadata["robot_manifest_sha256"],
         params=params, dimensions=dimensions, action_dim=action_dim, checkpoint=path.name,
         control=control_config(metadata["control_dt_s"]), residual_mean=True, seed=args.seed))
    print(f"Deployment: {output}")


def main():
    if "--train-local-imitators" in sys.argv:
        sys.argv.remove("--train-local-imitators")
        return legacy_main()
    from .training import training_arguments, train_residual
    parser = argparse.ArgumentParser(description="Train a residual on mjlab using frozen generalist imitators")
    training_arguments(parser, "residual")
    train_residual(parser.parse_args())


if __name__ == "__main__":
    main()
