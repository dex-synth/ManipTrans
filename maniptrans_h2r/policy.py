"""Native ManipTrans networks and normalizers, independent of a simulator."""

from copy import deepcopy

import gym
import numpy as np
import torch
from rl_games.algos_torch import model_builder

from lib.rl.models import ModelA2CContinuousLogStd
from lib.rl.network_builder import DictObsBuilder
from lib.rl.network_builder_residual_sh import ResRHDictObsBuilder, ResLHDictObsBuilder
from lib.rl.network_builder_residual_bih import ResBiHDictObsBuilder
from lib.rl.res_models import ModelA2CContinuousLogStdResRH, ModelA2CContinuousLogStdResLH, ModelA2CContinuousLogStdResBiH


def register():
    for name, network, model in (
        ("base", DictObsBuilder, ModelA2CContinuousLogStd),
        ("right", ResRHDictObsBuilder, ModelA2CContinuousLogStdResRH),
        ("left", ResLHDictObsBuilder, ModelA2CContinuousLogStdResLH),
        ("bimanual", ResBiHDictObsBuilder, ModelA2CContinuousLogStdResBiH),
    ):
        model_builder.register_network(f"h2r_{name}", network)
        model_builder.register_model(f"h2r_{name}", model)


def network_config(dimensions, small=False):
    return dict(name="h2r_base", separate=False, space={"continuous": {
        "mu_activation": "None", "sigma_activation": "None", "mu_init": {"name": "default"},
        "sigma_init": {"name": "const_initializer", "val": 0}, "fixed_sigma": True}},
        dict_feature_encoder={"_target_": "lib.nn.features.SimpleFeatureFusion",
            "extractors": {k: {"_target_": "lib.nn.features.Identity", "input_dim": n} for k, n in dimensions.items()},
            "hidden_depth": 3, "hidden_dim": 64 if small else 512, "output_dim": 32 if small else 256,
            "activation": "swish", "add_input_activation": False, "add_output_activation": False},
        mlp={"units": [64, 32] if small else [256, 512, 128, 64], "activation": "elu", "d2rl": False,
             "initializer": {"name": "default"}, "regularizer": {"name": "None"}})


def parameters(dimensions, base_dimensions, sides, bases, action_dim, small=False):
    network = network_config(dimensions, small)
    model = {"name": "h2r_base"}
    if bases:
        if set(bases) != set(sides):
            raise ValueError("Residual policy requires every trained imitator")
        variant = "bimanual" if len(sides) == 2 else sides[0]
        network["name"] = model["name"] = f"h2r_{variant}"
        network["base_model"] = network_config(base_dimensions, small)
        network["base_model"]["action_size"] = action_dim // len(sides)
        network["base_model"].update(rh_checkpoint=bases.get("right"), lh_checkpoint=bases.get("left"))
        model.update(base_model_obs_shape=base_dimensions, rh_base_model_checkpoint=bases.get("right"),
                     lh_base_model_checkpoint=bases.get("left"))
    return dict(algo={"name": "ppo"}, model=model, network=network)


def build(params, dimensions, action_dim, device):
    register()
    space = gym.spaces.Dict({k: gym.spaces.Box(-5, 5, (n,), np.float32) for k, n in dimensions.items()})
    config = dict(actions_num=action_dim, input_shape=space, num_seqs=1, value_size=1,
                  normalize_value=True, normalize_input=True, normalize_input_excluded_keys=[],
                  use_pid_control=False, **params)
    return model_builder.ModelBuilder().load(params).build(config).to(device)


def resolve_bases(params, root):
    params = deepcopy(params)
    for side, prefix in (("right", "rh"), ("left", "lh")):
        if f"{prefix}_base_model_checkpoint" in params["model"] and params["model"][f"{prefix}_base_model_checkpoint"]:
            path = str(root / params["model"][f"{prefix}_base_model_checkpoint"])
            params["model"][f"{prefix}_base_model_checkpoint"] = path
            params["network"]["base_model"][f"{prefix}_checkpoint"] = path
    return params


class Policy:
    def __init__(self, model, action_dim, seeds, device, residual_mean=True):
        self.model, self.action_dim, self.device = model.eval(), action_dim, device
        self.residual_mean = residual_mean
        self.generators = {key: torch.Generator(device=device).manual_seed(seed) for key, seed in seeds.items()}

    @torch.no_grad()
    def __call__(self, obs, episode_ids):
        noise = torch.stack([torch.randn(self.action_dim, generator=self.generators[e], device=self.device)
                             for e in episode_ids])
        result = self.model({"is_train": False, "obs": dict(obs), "base_noise": noise})
        action = result["mus"]
        if not self.residual_mean:
            residual_noise = torch.stack([torch.randn(self.action_dim, generator=self.generators[e], device=self.device)
                                         for e in episode_ids])
            action = action + result["sigmas"]*residual_noise
        if "base_actions" not in result:
            raise ValueError("Deployment requires a complete residual policy")
        # Exact native sigmoid soft clamp for [-1, 1].
        return -1 + 2*torch.sigmoid(2*torch.cat([result["base_actions"], action], -1))
