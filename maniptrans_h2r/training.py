"""Native PPO orchestration with versioned generalist bases and resumable epochs."""

import argparse
from copy import deepcopy
import importlib.metadata
from pathlib import Path
import random
import shutil
import subprocess

import numpy as np
import torch
import yaml
from rl_games.common.algo_observer import AlgoObserver
from rl_games.common.tr_helpers import DefaultRewardsShaper

from lib.rl.agent import PPOAgent
from lib.rl.numerics import enable_checks, require_finite
from .buffers import CAPACITY_KEYS, SimulationOverflow, capacities
from .control import CONTROL
from .io import digest, load_artifact, read_json, seal, sha256, write_json
from .policy import parameters, register

TRAINING_REVISION = 7

def training_arguments(parser, stage):
    defaults = yaml.safe_load((Path(__file__).parent / "config" / f"{stage}.yaml").read_text())
    parser.set_defaults(stage=stage, recipe=defaults)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    for name in ("worlds", "epochs", "horizon", "mini_epochs", "seed", "save_every", "diagnostic_every"):
        parser.add_argument("--"+name.replace("_", "-"), type=int, default=defaults[name])
    for name in CAPACITY_KEYS:
        parser.add_argument("--"+name.replace("_", "-"), type=int, default=defaults[name],
                            help="Explicit MJWarp allocation limit; physics and learning are unchanged")
    parser.add_argument("--small-network", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--resume", type=Path, help="Trusted local resume.pth at a complete PPO epoch")
    parser.add_argument("--allow-capacity-increase", action="store_true",
                        help="Allow only increased contact/constraint buffers when resuming")
    parser.add_argument("--resume-config", type=Path,
                        help="Original resolved_config.yaml for a legacy checkpoint without an embedded recipe")
    if stage == "residual":
        parser.add_argument("--imitator-right")
        parser.add_argument("--imitator-left")
        parser.add_argument("--allow-smoke-imitator", action="store_true")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def frozen_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "base_model" in k}


def recipe_identity(recipe):
    return {k: v for k, v in recipe.items() if k not in {"epochs", "save_every", "diagnostic_every"}}


def training_capacities(args):
    if not args.resume and (args.allow_capacity_increase or args.resume_config):
        raise ValueError("Resume options require --resume")
    return capacities({key: getattr(args, key) for key in CAPACITY_KEYS})


def capacity_resume_changes(old_binding, binding, old_recipe, recipe):
    """Verify the old recipe hash before allowing allocation-only increases."""
    if old_recipe is None or digest(recipe_identity(old_recipe)) != old_binding["recipe_id"]:
        raise ValueError("Capacity resume requires the checkpoint's hash-matching original recipe")
    if digest(recipe_identity(recipe)) != binding["recipe_id"]:
        raise ValueError("Current resume recipe hash differs")
    if {k: v for k, v in old_binding.items() if k != "recipe_id"} != {k: v for k, v in binding.items() if k != "recipe_id"}:
        raise ValueError("Resume task/corpus/base binding differs")
    old, new = recipe_identity(old_recipe), recipe_identity(recipe)
    if old.keys() != new.keys():
        raise ValueError("Resume recipe fields differ")
    before, after = capacities(old), capacities(new)
    changed = {k for k in old if old[k] != new[k]}
    if changed - set(CAPACITY_KEYS) or any(after[k] < before[k] for k in CAPACITY_KEYS):
        raise ValueError("Resume permits only contact/constraint capacity increases")
    return {k: dict(before=before[k], after=after[k]) for k in sorted(changed)}


class ReproductionAgent(PPOAgent):
    def __init__(self, params, output, recipe, binding, diagnostic=None):
        super().__init__("run", params)
        enable_checks(self.model)
        # Native optimizer includes frozen tensors; keep only trainable residual parameters.
        self.optimizer = torch.optim.Adam((p for p in self.model.parameters() if p.requires_grad), lr=self.last_lr, eps=1e-8)
        self.output, self.recipe, self.binding = output, recipe, binding
        self.diagnostic_callback = diagnostic
        self.last_diagnostic = -1
        self.last_final_diagnostic = -1
        self.restored_best = None
        self.frozen_before = frozen_state(self.model)

    def trancate_gradients_and_step(self):
        require_finite({n: p.grad for n, p in self.model.named_parameters()}, "gradients")
        super().trancate_gradients_and_step()
        require_finite({n: p for n, p in self.model.named_parameters() if p.requires_grad}, "updated parameters")

    def save(self, filename):
        require_finite(self.model.state_dict(), "checkpoint")
        return super().save(filename)

    def verify_frozen(self):
        if any(not torch.equal(v, self.model.state_dict()[k].detach().cpu()) for k, v in self.frozen_before.items()):
            raise RuntimeError("Frozen base parameters or normalization buffers changed")

    def get_full_state_weights(self):
        state = super().get_full_state_weights()
        state["h2r"] = dict(binding=self.binding, recipe=deepcopy(self.recipe), common_steps=self.vec_env.common_steps,
            sampler=self.vec_env.sampler_state(), python_rng=random.getstate(), numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            scheduler=deepcopy(vars(self.scheduler)), last_lr=self.last_lr, entropy_coef=self.entropy_coef,
            last_diagnostic=self.last_diagnostic, last_final_diagnostic=self.last_final_diagnostic,
            world_state_restarted=True)
        return state

    def restore_local(self, path, *, allow_capacity_increase=False, resume_config=None):
        state = torch.load(path, map_location=self.ppo_device, weights_only=False)
        extra = state["h2r"]
        old_recipe = extra.get("recipe")
        if old_recipe is not None and digest(recipe_identity(old_recipe)) != extra["binding"]["recipe_id"]:
            raise ValueError("Embedded resume recipe does not match checkpoint recipe hash")
        if resume_config is not None:
            supplied = yaml.safe_load(Path(resume_config).read_text())
            if digest(recipe_identity(supplied)) != extra["binding"]["recipe_id"]:
                raise ValueError("Original resume config does not match checkpoint recipe hash")
            if old_recipe is not None and recipe_identity(old_recipe) != recipe_identity(supplied):
                raise ValueError("Supplied and embedded resume recipes differ")
            old_recipe = supplied
        changes = {}
        if extra["binding"] != self.binding:
            if not allow_capacity_increase:
                raise ValueError("Resume task/corpus/base/recipe binding differs")
            changes = capacity_resume_changes(extra["binding"], self.binding, old_recipe, self.recipe)
        self.set_full_state_weights(state)
        require_finite(self.model.state_dict(), "restored model")
        for values in self.optimizer.state.values():
            require_finite({k: v for k, v in values.items() if isinstance(v, torch.Tensor)}, "restored optimizer")
        self.restored_best = self.last_mean_rewards
        self.scheduler.__dict__.update(extra["scheduler"])
        self.last_lr, self.entropy_coef = extra["last_lr"], extra["entropy_coef"]
        self.update_lr(self.last_lr)
        self.vec_env.common_steps = extra["common_steps"]
        self.vec_env.load_sampler_state(extra["sampler"])
        random.setstate(extra["python_rng"])
        np.random.set_state(extra["numpy_rng"])
        torch.set_rng_state(extra["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in extra["cuda_rng"]])
        self.last_diagnostic = extra["last_diagnostic"]
        self.last_final_diagnostic = extra["last_final_diagnostic"]
        self.verify_frozen()
        write_json(self.output / "resume.json", dict(checkpoint=str(Path(path).resolve()),
            checkpoint_sha256=sha256(path), epoch=self.epoch_num, frame=self.frame,
            old_binding=extra["binding"], binding=self.binding, capacity_changes=changes,
            world_state_restarted=True))

    def env_reset(self):
        obs = super().env_reset()
        if self.restored_best is not None:
            self.last_mean_rewards = self.restored_best
            self.restored_best = None
        return obs

    def diagnostic(self, *, final=False):
        mean = self.last_diagnostic != self.epoch_num
        final = final and self.last_final_diagnostic != self.epoch_num
        if self.diagnostic_callback is None or not (mean or final):
            return
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
        mode = self.model.training
        try:
            self.diagnostic_callback(self.model, self.epoch_num, mean=mean, final=final)
        finally:
            random.setstate(rng[0]); np.random.set_state(rng[1])
            torch.set_rng_state(rng[2]); torch.cuda.set_rng_state_all(rng[3])
            self.model.train(mode)
        self.last_diagnostic = self.epoch_num
        if final:
            self.last_final_diagnostic = self.epoch_num

    def update_epoch(self):
        # Parent loop has completed frame/best-reward bookkeeping at this point.
        if self.epoch_num and self.epoch_num % self.recipe["save_every"] == 0:
            self.save_product(f"epoch-{self.epoch_num}")
        period = self.recipe["diagnostic_every"]
        if period and self.epoch_num % period == 0:
            self.diagnostic()
        return super().update_epoch()

    def save_product(self, name):
        self.verify_frozen()
        require_finite(self.model.state_dict(), "checkpoint")
        torch.save({"model": self.model.state_dict()}, self.output / f"{name}.pth")
        torch.save(self.get_full_state_weights(), self.output / "resume.pth")


def run_training(env, params, args, output, prepared_meta, base_ids=None, diagnostic=None):
    source_root = Path(__file__).resolve().parent.parent
    source_files = sorted({*source_root.glob("maniptrans_h2r/**/*.py"), *source_root.glob("maniptrans_h2r/config/*.yaml"),
                           *source_root.glob("lib/rl/**/*.py"), *source_root.glob("lib/utils/*.py"),
                           *source_root.glob("main/cfg/**/*.yaml"), source_root / "requirements-h2r.lock"})
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source_root, capture_output=True, text=True, check=False)
    write_json(output / "source.json", dict(commit=revision.stdout.strip() or None,
        files={str(p.relative_to(source_root)): sha256(p) for p in source_files},
        gpu=torch.cuda.get_device_name(args.device), cuda=torch.version.cuda))
    recipe = dict(args.recipe)
    recipe.update(env.buffer_capacities)
    for key in ("worlds", "epochs", "horizon", "mini_epochs", "seed", "save_every", "diagnostic_every"):
        recipe[key] = getattr(args, key)
    recipe.update(small_network=args.small_network, partial=prepared_meta.get("partial", False),
                  training_framework="mjlab", physics_engine="mujoco_warp", training_revision=TRAINING_REVISION, base_sampling="native-resample",
                  control=dict(CONTROL),
                  native_config_source="main/cfg/config.yaml + task and rl_train stage defaults",
                  budget_source="H2R explicit limits")
    if args.stage == "imitator":
        recipe["wrist_rsi"] = dict(env.wrist_rsi_recipe)
    batch = args.worlds*args.horizon
    minibatch = min(1024, batch) if args.small_network else 1024
    if min(args.worlds, args.epochs, args.horizon, args.mini_epochs, args.save_every) < 1 or batch % minibatch or args.diagnostic_every < 0:
        raise ValueError("Require positive sizes and a batch divisible by minibatch (1024 outside smoke mode)")
    recipe["minibatch"] = minibatch
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(recipe, sort_keys=True))
    capacities(recipe)
    binding = dict(prepared_id=prepared_meta["id"], recipe_id=digest(recipe_identity(recipe)), bases=base_ids or {})
    saved_params = deepcopy(params)
    config = dict(name=args.stage, full_experiment_name=args.stage, env_name="h2r_mjlab", vec_env=env,
        env_info=env.get_env_info(), device=args.device, num_actors=args.worlds, train_dir=str(output / "training"),
        features={"observer": AlgoObserver()}, reward_shaper=DefaultRewardsShaper(scale_value=1),
        use_pid_control=False, ppo=True, normalize_input=True, normalize_input_excluded_keys=[],
        normalize_value=True, normalize_advantage=True, value_bootstrap=True,
        gamma=recipe["gamma"], tau=recipe["tau"], learning_rate=recipe["learning_rate"], lr_schedule=recipe["lr_schedule"],
        warmup_steps=recipe.get("warmup_steps", 10), start_lr=recipe.get("start_lr", 1e-6),
        schedule_type="standard", kl_threshold=recipe["kl_threshold"], score_to_win=recipe["score_to_win"],
        max_epochs=args.epochs, early_stop_epochs=recipe["early_stop_epochs"], save_best_after=recipe["save_best_after"], save_frequency=0,
        print_stats=True, grad_norm=1., entropy_coef=0., truncate_grads=True, e_clip=.2,
        error_if_nonfinite_gradients=True,
        horizon_length=args.horizon, minibatch_size=minibatch, mini_epochs=args.mini_epochs,
        critic_coef=4, clip_value=True, bounds_loss_coef=.0001, use_soft_clamp=True,
        store_base_actions=False, mixed_precision=False, weight_decay=0.)
    write_json(output / "ppo_config.json", {k: v for k, v in config.items()
        if k not in {"vec_env", "env_info", "features", "reward_shaper"}})
    params["config"] = config
    register()
    agent = ReproductionAgent(params, output, recipe, binding, diagnostic)
    if args.resume:
        agent.restore_local(args.resume, allow_capacity_increase=args.allow_capacity_increase,
                            resume_config=args.resume_config)
        if agent.epoch_num >= args.epochs:
            raise ValueError("Resume epoch must be below requested total --epochs")
    before = {k: v.detach().clone() for k, v in agent.model.named_parameters() if v.requires_grad}
    try:
        agent.train()
        if not any(not torch.equal(v, dict(agent.model.named_parameters())[k]) for k, v in before.items()):
            raise RuntimeError("No trainable parameter changed")
        agent.diagnostic(final=True)
        agent.save_product("model")
    except Exception as exc:
        failure = dict(type=type(exc).__name__, error=str(exc), epoch=agent.epoch_num, frame=agent.frame)
        if isinstance(exc, SimulationOverflow):
            failure["overflow"] = exc.report
        write_json(output / "failure.json", failure)
        raise
    finally:
        agent.writer.close()
    if env.curriculum is not None:
        env.audit["curriculum"] = env.curriculum.audit()
    write_json(output / "physics_audit.json", env.audit)
    write_json(output / "training.json", dict(epoch=agent.epoch_num, transitions=agent.frame,
        common_control_steps=env.common_steps, scale_factor=env.scale,
        stop_reason=("native_score_to_win" if float(agent.last_mean_rewards) > recipe["score_to_win"]
                     else "epoch_budget" if agent.epoch_num >= args.epochs else "native_early_stop"),
        frozen_state_unchanged=True, world_state_restarted=bool(args.resume),
        versions={p: importlib.metadata.version(p) for p in ("mjlab", "mujoco", "mujoco-warp", "warp-lang", "torch", "rl-games")}))
    if hasattr(env, "motion"):
        write_json(output / "sampling.json", dict(visits=env.motion.visits.tolist(), transitions=env.motion.transitions.tolist(),
                                                  wrist_rsi=env.wrist_rsi_stats))
    return saved_params, recipe


def train_imitator(args):
    from .imitator_env import ImitatorEnv
    from .diagnostic import diagnostic_callback
    torch.set_num_threads(4)
    limits = training_capacities(args)
    seed_all(args.seed)
    prepared, metadata = load_artifact(args.prepared)
    output = Path(args.output).resolve()
    if output.is_relative_to(prepared):
        raise ValueError("Output must be outside prepared inputs")
    output.mkdir(parents=True, exist_ok=False)
    env = ImitatorEnv(prepared, args.worlds, args.device, args.seed, args.allow_incomplete, buffer_capacities=limits)
    dimensions = {k: s.shape[0] for k, s in env.observation_space.spaces.items()}
    params = parameters(dimensions, dimensions, env.sides, {}, env.action_dim, args.small_network)
    saved, recipe = run_training(env, params, args, output, metadata,
        diagnostic=diagnostic_callback(prepared, output, args.device, args.allow_incomplete, buffer_capacities=limits))
    for name in ("robot_mapping.json", "compatibility.json", "motion_manifest.json", "diagnostic_manifest.json", "robot_scene.xml"):
        shutil.copy2(prepared / name, output / name)
    shutil.copytree(prepared / "assets", output / "assets")
    seal(output, "imitator.json", dict(format="maniptrans-h2r-imitator-v1", side=metadata["side"],
        prepared_id=metadata["id"], corpus_id=metadata["corpus_id"], compatibility=metadata["compatibility"],
        control=CONTROL,
        smoke=bool(args.small_network or metadata["partial"]), params=saved, dimensions=dimensions,
        action_dim=env.action_dim, checkpoint="model.pth", recipe=recipe))
    print(f"Generalist imitator: {output}")


def train_residual(args):
    from .residual_env import ResidualEnv
    from .diagnostic import residual_diagnostic_callback
    torch.set_num_threads(4)
    limits = training_capacities(args)
    seed_all(args.seed)
    prepared, metadata = load_artifact(args.prepared)
    robot = read_json(prepared / "robot_mapping.json")
    bases, manifests = {}, {}
    for side in robot["sides"]:
        path = getattr(args, f"imitator_{side}")
        if not path:
            raise ValueError(f"Residual training requires --imitator-{side}; no local base is trained implicitly")
        root, manifest = load_artifact(path, "imitator.json")
        if manifest["format"] != "maniptrans-h2r-imitator-v1" or manifest["side"] != side:
            raise ValueError("Wrong imitator format or hand side")
        if manifest.get("recipe", {}).get("training_revision") != TRAINING_REVISION:
            raise ValueError("Retrain the base with the current wrist control and training revision")
        if manifest["smoke"] and not args.allow_smoke_imitator:
            raise ValueError("Smoke base requires --allow-smoke-imitator")
        if manifest["compatibility"] != metadata["compatibility"][side] or manifest["control"] != CONTROL:
            raise ValueError(f"Incompatible {side} base physics/reference/control signature")
        bases[side], manifests[side] = root, manifest
    output = Path(args.output).resolve()
    if output.is_relative_to(prepared):
        raise ValueError("Output must be outside prepared inputs")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(prepared, output / "prepared")
    checkpoints = {}
    for side, root in bases.items():
        target = output / f"{side}-base.pth"
        shutil.copy2(root / manifests[side]["checkpoint"], target)
        write_json(output / f"{side}-base.json", manifests[side])
        checkpoints[side] = str(target)
    env = ResidualEnv(output / "prepared", args.worlds, args.device, args.seed, buffer_capacities=limits)
    dimensions = {k: s.shape[0] for k, s in env.observation_space.spaces.items()}
    for manifest in manifests.values():
        if manifest["dimensions"] != env.features.base_dimensions:
            raise ValueError("Base feature order/dimensions differ")
    params = parameters(dimensions, env.features.base_dimensions, env.sides, checkpoints, env.action_dim, args.small_network)
    continuous = params["network"]["space"]["continuous"]
    continuous["mu_init"] = dict(name="glorot_normal_initializer", gain=.01)
    continuous["sigma_init"] = dict(name="const_initializer", val=-1)
    first = manifests[env.sides[0]]["params"]["network"]
    params["network"]["base_model"].update(deepcopy(first))
    for side, prefix in (("right", "rh"), ("left", "lh")):
        if side in manifests:
            params["network"][f"{prefix}_base_model"] = deepcopy(manifests[side]["params"]["network"])
    saved, recipe = run_training(env, params, args, output, metadata, {s: m["id"] for s, m in manifests.items()},
        diagnostic=residual_diagnostic_callback(output / "prepared", output, args.device, buffer_capacities=limits))
    for side, prefix in (("right", "rh"), ("left", "lh")):
        if side in bases:
            saved["model"][f"{prefix}_base_model_checkpoint"] = f"{side}-base.pth"
            saved["network"]["base_model"][f"{prefix}_checkpoint"] = f"{side}-base.pth"
    seal(output, "deployment.json", dict(format="maniptrans-h2r-deployment-v2", method="maniptrans-mjlab",
        controller=CONTROL["schema"], prepared_id=metadata["id"], task_id=metadata["task_id"],
        bundle_id=metadata["bundle_id"], robot_manifest_sha256=metadata["robot_manifest_sha256"],
        compatibility=metadata["compatibility"], params=saved, dimensions=dimensions, action_dim=env.action_dim,
        checkpoint="model.pth", control=CONTROL, residual_mean=True, recipe=recipe,
        bases={s: m["id"] for s, m in manifests.items()}))
    print(f"Residual deployment: {output}")
