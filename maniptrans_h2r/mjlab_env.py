"""mjlab Simulation lifecycle and the native ManipTrans PPO vector interface."""

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import gym
import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
import yaml
from mjlab.sim import Simulation, SimulationCfg
from mjlab.sim.sim import MujocoCfg

from .control import CONTROL, PositionControl
from .buffers import capacities, check_overflow
from .observations import Features
from .physics import CAPACITY, OPTION_FIELDS, equal, gpu_audit, sensor_columns

def simulation_config(model, stage, buffer_capacities=None):
    cfg = MujocoCfg()
    for name in ("timestep", "impratio", "iterations", "tolerance", "ls_iterations", "ls_tolerance", "ccd_iterations"):
        setattr(cfg, name, getattr(model.opt, name))
    cfg.gravity = tuple(model.opt.gravity)
    for name, enum, prefix in (("integrator", mujoco.mjtIntegrator, "mjINT_"),
                               ("solver", mujoco.mjtSolver, "mjSOL_"),
                               ("cone", mujoco.mjtCone, "mjCONE_"),
                               ("jacobian", mujoco.mjtJacobian, "mjJAC_")):
        value = enum(getattr(model.opt, name)).name.removeprefix(prefix).lower()
        setattr(cfg, name, value)
    cfg.disableflags, cfg.enableflags = (), ()  # Existing bits stay on the loaded model.
    recipe = yaml.safe_load((Path(__file__).parent / "config" / f"{stage}.yaml").read_text())
    recipe = capacities(recipe if buffer_capacities is None else buffer_capacities)
    return SimulationCfg(mujoco=cfg, nconmax=recipe["contact_capacity"],
                         njmax=recipe["constraint_capacity"], contact_sensor_maxmatch=CAPACITY+1)


def hand_model(path, robot):
    tree = ET.parse(path)
    for node in tree.iter():
        if "file" in node.attrib:
            node.set("file", str((path.parent / node.get("file")).resolve()))
    sensors = tree.getroot().find("sensor")
    if sensors is None:
        sensors = ET.SubElement(tree.getroot(), "sensor")
    bodies = [b for hand in robot["hands"].values() for b in hand["bodies"]]
    for i, body in enumerate(bodies):
        for tag in ("framepos", "framequat", "framelinvel", "frameangvel"):
            ET.SubElement(sensors, tag, name=f"mt_{tag}_{i}", objtype="xbody", objname=body)
    return mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding="unicode")), bodies


class MjlabVecEnv:
    def __init__(self, model, robot, bodies, pairs, count, device, metadata, stage, buffer_capacities=None):
        if not str(device).startswith("cuda"):
            raise ValueError("mjlab training requires an explicit CUDA device")
        self.cpu, self.robot, self.bodies, self.pairs = model, robot, bodies, pairs
        self.count, self.device, self.metadata, self.stage = count, device, metadata, stage
        self.sides = robot["sides"]
        self.env = self
        self.control_dt_s, self.decimation = 1/60, 2
        torch.cuda.set_device(device)
        wp.init()
        wp.set_device(device)
        self.torch_stream = torch.cuda.current_stream(device)
        if self.torch_stream == torch.cuda.default_stream(device):
            # CUDA graph capture cannot use the legacy default stream.
            self.torch_stream = torch.cuda.Stream(device=device)
            self.torch_stream.wait_stream(torch.cuda.current_stream(device))
            torch.cuda.set_stream(self.torch_stream)
        # Reuse the wrapper: deleting a second wrapper for the same Torch stream
        # unregisters the live handle in Warp when a diagnostic env is collected.
        self.warp_stream = wp.get_stream(device)
        if self.warp_stream.cuda_stream != self.torch_stream.cuda_stream:
            self.warp_stream = wp.stream_from_torch(self.torch_stream)
        wp.set_stream(self.warp_stream)
        options = {n: np.asarray(getattr(model.opt, n)).copy() for n in OPTION_FIELDS}
        self.sim = Simulation(num_envs=count, cfg=simulation_config(model, stage, buffer_capacities), model=model, device=device)
        self.buffer_capacities = dict(contact_capacity=self.sim.cfg.nconmax, constraint_capacity=self.sim.cfg.njmax)
        for n, value in options.items():
            equal(getattr(self.sim.mj_model.opt, n), value, f"mjlab.option.{n}")
        self.model, self.data = self.sim.wp_model, self.sim.wp_data
        self.model.opt.tolerance.fill_(float(model.opt.tolerance))
        self.sim.create_graph()
        self.audit = gpu_audit(model, self.model, {"task_id": metadata.get("task_id", "single-hand")}, metadata["scene_sha256"])
        self.audit["training_framework"] = "mjlab"
        self.audit["capacity"] = dict(contacts_per_world=self.sim.cfg.nconmax, constraints_per_world=self.sim.cfg.njmax,
                                     contact_sensor=CAPACITY)
        self.views = {n: wp.to_torch(getattr(self.data, n)) for n in
                      ("qpos", "qvel", "ctrl", "act", "time", "qacc_warmstart", "sensordata", "qfrc_actuator")}
        self.control = PositionControl(robot, count, device, CONTROL)
        self.features = Features(robot, metadata.get("part_names", []), float(np.linalg.norm(model.opt.gravity)))
        self.tick = torch.zeros(count, dtype=torch.long, device=device)
        self.elapsed_ticks = torch.zeros_like(self.tick)
        self.completed_controls = torch.zeros_like(self.tick)
        self.curriculum = None
        self.done = torch.ones(count, dtype=torch.bool, device=device)
        self.success_buf = torch.zeros(count, device=device)
        self.failure_buf = torch.zeros_like(self.success_buf)
        self.common_steps = 0
        self.fixed_scale = None
        self.tip_force = {s: torch.zeros((count, 5, 3), device=device) for s in self.sides}
        self.contact_history = {s: torch.ones((count, 3, 5), dtype=torch.bool, device=device) for s in self.sides}
        self.contact_columns = torch.tensor([list(range(*sensor_columns(model, name).indices(model.nsensordata)))
                                             for _, _, name in pairs], device=device, dtype=torch.long)
        self.contact_destinations = torch.tensor([self.sides.index(s)*5+t for s, t, _ in pairs], device=device, dtype=torch.long)
        self.action_dim = len(robot["joint_names"])
        self.object_qids, self.object_vids = [], []

    def finish_init(self):
        obs = self.reset()
        self.observation_space = gym.spaces.Dict({k: gym.spaces.Box(-5, 5, (v.shape[-1],), np.float32) for k, v in obs.items()})
        self.action_space = gym.spaces.Box(-1, 1, (self.action_dim,), np.float32)

    @property
    def scale(self):
        denominator = 128000 if self.stage == "imitator" else 3200
        return self.fixed_scale if self.fixed_scale is not None else .7+.3*(2*math.e)**(-self.common_steps/denominator)

    def contacts(self):
        if not self.pairs:
            return {s: torch.zeros_like(v) for s, v in self.tip_force.items()}
        values = self.views["sensordata"][:, self.contact_columns].reshape(self.count, len(self.pairs), CAPACITY+1, 14)
        if not torch.isfinite(values).all() or torch.any(values[:, :, -1, 0] != 0):
            raise RuntimeError("Invalid or overflowing contact sensor")
        force, normal, tangent = values[..., 1:4], values[..., 8:11], values[..., 11:14]
        world = force[..., :1]*normal+force[..., 1:2]*tangent+force[..., 2:]*torch.cross(normal, tangent, dim=-1)
        result = torch.zeros((self.count, len(self.sides)*5, 3), device=self.device)
        result.index_add_(1, self.contact_destinations, -(world*(values[..., :1] != 0)).sum(2))
        return {s: result[:, 5*i:5*i+5] for i, s in enumerate(self.sides)}

    def state(self):
        # Refresh only post-integration kinematics, preserving the actual last solve.
        mjw.kinematics(self.model, self.data)
        mjw.com_pos(self.model, self.data)
        mjw.com_vel(self.model, self.data)
        mjw.sensor_pos(self.model, self.data)
        mjw.sensor_vel(self.model, self.data)
        bodies = {b: torch.cat([self.views["sensordata"][:, sensor_columns(self.cpu, f"mt_{tag}_{i}")]
                   for tag in ("framepos", "framequat", "framelinvel", "frameangvel")], -1) for i, b in enumerate(self.bodies)}
        objects = [bodies[b] for b in self.metadata.get("object_bodies", [])]
        return dict(robot_q=self.views["qpos"][:, self.robot["qpos_indices"]],
            robot_dq=self.views["qvel"][:, self.robot["qvel_indices"]], bodies=bodies,
            objects=torch.stack(objects, 1) if objects else torch.empty((self.count, 0, 13), device=self.device),
            object_q=self.views["qpos"][:, self.object_qids], object_dq=self.views["qvel"][:, self.object_vids], tip_force=self.tip_force)

    def observe(self):
        return self.features(self.state(), self.target(1), self.sides, self.stage)

    def reset(self):
        self.done[:] = True
        return self.reset_done()[0]

    def reset_done(self):
        ids = self.done.nonzero().flatten()
        if len(ids):
            unaffected = ~self.done
            saved = {k: v[unaffected].clone() for k, v in self.views.items()}
            if self.curriculum is not None:
                self.curriculum.reset(ids.cpu().numpy(), self.common_steps)
            self.sim.reset(ids)
            self.reset_states(ids)
            self.views["ctrl"][ids[:, None], torch.as_tensor(self.robot["actuator_indices"], device=self.device)] = self.views["qpos"][ids][:, self.robot["qpos_indices"]][:, self.robot["ctrl_joint_indices"]]
            self.sim.forward()
            check_overflow(self.data, "reset")
            for k, value in saved.items():
                self.views[k][unaffected] = value
            contacts = self.contacts()
            for s in self.sides:
                self.tip_force[s][ids] = contacts[s][ids]
                self.contact_history[s][ids] = True
            self.control.reset(ids, self.views["qpos"][ids][:, self.robot["qpos_indices"]])
            self.elapsed_ticks[ids] = 0
            self.completed_controls[ids] = 0
            self.done[ids] = False
        return self.observe(), ids

    @torch.no_grad()
    def step(self, actions):
        if self.done.any():
            raise RuntimeError("Call reset_done before stepping completed environments")
        target = self.target(1)
        positions = self.views["qpos"][:, self.robot["qpos_indices"]]
        commands = self.control.decode(actions, target["robot_q"], self.sides, self.stage == "residual", positions=positions)
        self.views["ctrl"][:, self.robot["actuator_indices"]] = commands[:, self.robot["ctrl_joint_indices"]]
        steps = self.remaining_ticks().clamp(max=2)
        power = torch.zeros_like(commands)
        for substep in range(int(steps.max())):
            active = steps > substep
            inactive = ~active
            saved = {k: v[inactive].clone() for k, v in self.views.items()}
            self.sim.step()
            check_overflow(self.data, "step")
            power[active] = (self.views["qfrc_actuator"][:, self.robot["qvel_indices"]]
                             * self.views["qvel"][:, self.robot["qvel_indices"]])[active]
            forces = self.contacts()
            for s in self.sides:
                self.tip_force[s][active] = forces[s][active]
            for k, values in saved.items():
                self.views[k][inactive] = values
        self.tick += steps
        self.elapsed_ticks += steps
        self.advance_reference()
        self.common_steps += 1
        if self.curriculum is not None:
            self.curriculum.advance()
        for s in self.sides:
            self.contact_history[s] = torch.cat([self.contact_history[s][:, 1:], (self.tip_force[s].norm(dim=-1)>0)[:, None]], dim=1)
        state = self.state()
        self.last_power = power
        reward, failure, details = self.reward(state, self.target(0), power, actions)
        self.completed_controls += 1
        if any(not torch.isfinite(self.views[k]).all() for k in ("qpos", "qvel")) or not torch.isfinite(reward).all():
            raise RuntimeError("Non-finite physical state or native reward")
        at_end = self.remaining_ticks() == 0
        self.done = failure | at_end
        self.success_buf, self.failure_buf = (at_end & ~failure).float(), failure.float()
        details["scale_factor"] = torch.full_like(reward, self.scale)
        obs = self.features(state, self.target(1), self.sides, self.stage)
        return obs, reward, self.done, dict(reward_dict=details, error_masks=torch.ones_like(reward), time_outs=torch.zeros_like(self.done))

    def advance_reference(self):
        pass

    def get_env_info(self):
        return dict(observation_space=self.observation_space, action_space=self.action_space, agents=1, value_size=1)

    def set_train_info(self, frame, agent):
        pass

    def get_env_state(self):
        return None

    def set_env_state(self, state):
        if state is not None:
            raise ValueError("Checkpoints cannot supply simulation state")
