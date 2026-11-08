"""MJWarp worlds using sealed task physics and the native PPO VecEnv interface."""

import gym
import math
import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from lib.rl.rewards import imitator_reward, manipulation_reward
from .control import PositionControl
from .io import load_artifact, load_arrays, load_task, read_json
from .observations import Features, Reference, xyzw
from .physics import CAPACITY, gpu_audit, load_model, sensor_columns
from .robot import initial_qpos, weight_indices


class MJWarpEnv:
    def __init__(self, prepared, stage, sides, count, device, control):
        if not device.startswith("cuda"):
            raise ValueError("H2R training requires MJWarp on CUDA")
        self.path, self.metadata = load_artifact(prepared)
        self.task_path, self.task = load_task(self.path / "task")
        self.robot = read_json(self.path / "robot_mapping.json")
        self.stage, self.sides, self.count, self.device = stage, sides, count, device
        self.env = self
        torch.cuda.set_device(device)
        wp.init()
        wp.set_device(device)
        self.torch_stream = torch.cuda.Stream(device=device)
        torch.cuda.set_stream(self.torch_stream)
        self.warp_stream = wp.stream_from_torch(self.torch_stream)
        wp.set_stream(self.warp_stream)
        self.cpu, self.bodies, self.pairs, scene_hash = load_model(self.task_path, self.task, self.robot)
        self.model = mjw.put_model(self.cpu)
        # MJWarp defaults to a float32 tolerance floor. Retain the sealed task value.
        self.model.opt.tolerance.fill_(float(self.cpu.opt.tolerance))
        self.model.opt.contact_sensor_maxmatch = CAPACITY + 1
        self.audit = gpu_audit(self.cpu, self.model, self.task, scene_hash)
        data = mujoco.MjData(self.cpu)
        self.initial = initial_qpos(self.cpu, self.task)
        self.reference = Reference(load_arrays(self.path / "reference.npz"), device)
        self.initial[self.robot["qpos_indices"]] = self.reference.values["robot_q"][0].cpu().numpy()
        data.qpos[:] = self.initial
        data.ctrl[self.robot["actuator_indices"]] = self.initial[self.robot["qpos_indices"]][self.robot["ctrl_joint_indices"]]
        mujoco.mj_forward(self.cpu, data)
        self.data = mjw.put_data(self.cpu, data, nworld=count, nconmax=2048, njmax=8192)
        mjw.step(self.model, self.data)  # Compile before capture; reset below discards this warmup.
        with wp.ScopedCapture(device=device) as capture:
            mjw.step(self.model, self.data)
        self.step_graph = capture.graph
        self.views = {k: wp.to_torch(getattr(self.data, k)) for k in
                      ("qpos", "qvel", "ctrl", "act", "time", "qacc_warmstart", "sensordata", "qfrc_actuator")}
        self.control = PositionControl(self.robot, count, device, control)
        self.features = Features(self.robot, self.metadata["part_names"], float(np.linalg.norm(self.cpu.opt.gravity)))
        self.tick = torch.zeros(count, dtype=torch.long, device=device)
        self.done = torch.zeros(count, dtype=torch.bool, device=device)
        self.success_buf = torch.zeros(count, device=device)
        self.failure_buf = torch.zeros(count, device=device)
        self.tip_force = {s: torch.zeros((count, 5, 3), device=device) for s in self.robot["sides"]}
        history_length = max(1, math.ceil((3/60) / self.metadata["control_dt_s"]))
        self.contact_history = {s: torch.ones((count, history_length, 5), dtype=torch.bool, device=device)
                                for s in self.robot["sides"]}
        self.contact_columns = torch.tensor([list(range(*sensor_columns(self.cpu, name).indices(self.cpu.nsensordata)))
                                             for _, _, name in self.pairs], device=device)
        self.contact_destinations = torch.tensor([self.robot["sides"].index(side)*5 + tip
                                                 for side, tip, _ in self.pairs], device=device)
        self.object_qids = [int(self.cpu.joint(n).qposadr[0]) for n in self.task["object"]["joints"].values()]
        self.object_vids = [int(self.cpu.joint(n).dofadr[0]) for n in self.task["object"]["joints"].values()]
        self.action_dim = sum(len(self.robot["hands"][s]["indices"]) for s in sides)
        obs = self.reset()
        self.observation_space = gym.spaces.Dict({k: gym.spaces.Box(-5, 5, (v.shape[-1],), np.float32)
                                                   for k, v in obs.items()})
        self.action_space = gym.spaces.Box(-1, 1, (self.action_dim,), np.float32)

    def get_env_info(self):
        return dict(observation_space=self.observation_space, action_space=self.action_space, agents=1, value_size=1)

    def _contacts(self):
        values = self.views["sensordata"][:, self.contact_columns].reshape(self.count, len(self.pairs), CAPACITY+1, 14)
        if not torch.isfinite(values).all() or torch.any(values[:, :, -1, 0] != 0):
            raise RuntimeError("Invalid or overflowing MJWarp contact sensor")
        force, normal, tangent = values[..., 1:4], values[..., 8:11], values[..., 11:14]
        world = force[..., :1]*normal + force[..., 1:2]*tangent + force[..., 2:]*torch.cross(normal, tangent, dim=-1)
        result = torch.zeros((self.count, len(self.robot["sides"])*5, 3), device=self.device)
        result.index_add_(1, self.contact_destinations, -(world*(values[..., :1] != 0)).sum(2))
        return {s: result[:, i*5:(i+1)*5] for i, s in enumerate(self.robot["sides"])}

    def state(self):
        # Refresh only kinematics: retain the previous solve's contact forces and warm start.
        mjw.kinematics(self.model, self.data)
        mjw.com_pos(self.model, self.data)
        mjw.com_vel(self.model, self.data)
        mjw.sensor_pos(self.model, self.data)
        mjw.sensor_vel(self.model, self.data)
        bodies = {b: torch.cat([self.views["sensordata"][:, sensor_columns(self.cpu, f"mt_{tag}_{i}")]
                                for tag in ("framepos", "framequat", "framelinvel", "frameangvel")], -1)
                  for i, b in enumerate(self.bodies)}
        return dict(robot_q=self.views["qpos"][:, self.robot["qpos_indices"]],
                    robot_dq=self.views["qvel"][:, self.robot["qvel_indices"]], bodies=bodies,
                    objects=torch.stack([bodies[b] for b in self.task["object"]["parts"].values()], 1),
                    object_q=self.views["qpos"][:, self.object_qids],
                    object_dq=self.views["qvel"][:, self.object_vids], tip_force=self.tip_force)

    def target(self, lookahead=0):
        return self.reference.sample((self.tick + lookahead).clamp(max=self.task["end_tick"])*self.task["dt_s"])

    def observe(self):
        return self.features(self.state(), self.target(self.task["control_ticks"]), self.sides, self.stage)

    def reset(self):
        self.done[:] = True
        return self.reset_done()[0]

    def reset_done(self):
        ids = self.done.nonzero().flatten()
        if len(ids):
            mask = self.done.clone()
            mjw.reset_data(self.model, self.data, wp.from_torch(mask))
            self.views["qpos"][ids] = torch.as_tensor(self.initial, dtype=torch.float32, device=self.device)
            q = self.views["qpos"][ids][:, self.robot["qpos_indices"]]
            self.views["ctrl"][ids[:, None], self.robot["actuator_indices"]] = q[:, self.robot["ctrl_joint_indices"]]
            self.control.reset(ids, q)
            self.tick[ids] = 0
            self.done[ids] = False
            # Forward is needed for tick-zero contacts; preserve unaffected histories.
            warmstart = self.views["qacc_warmstart"].clone()
            mjw.forward(self.model, self.data)
            self.views["qacc_warmstart"].copy_(warmstart)
            contacts = self._contacts()
            for side in self.tip_force:
                self.tip_force[side][ids] = contacts[side][ids]
                self.contact_history[side][ids] = True
        return self.observe(), ids

    @torch.no_grad()
    def step(self, actions):
        if self.done.any():
            raise RuntimeError("Call reset_done before stepping finished worlds")
        next_tick = (self.tick + self.task["control_ticks"]).clamp(max=self.task["end_tick"])
        targets = self.control.decode(actions, self.target(self.task["control_ticks"])["robot_q"],
                                      self.sides, self.stage == "residual")
        self.views["ctrl"][:, self.robot["actuator_indices"]] = targets[:, self.robot["ctrl_joint_indices"]]
        power = torch.zeros((self.count, len(self.robot["joint_names"])), device=self.device)
        steps = next_tick-self.tick
        for substep in range(int(steps.max())):
            inactive = steps <= substep
            saved = {k: self.views[k][inactive].clone() for k in ("qpos", "qvel", "act", "time", "qacc_warmstart")}
            velocity = self.views["qvel"][:, self.robot["qvel_indices"]].clone()
            wp.capture_launch(self.step_graph)
            if torch.any(wp.to_torch(self.data.overflow) != 0):
                raise RuntimeError("MJWarp constraint/contact capacity exceeded")
            force = self.views["qfrc_actuator"][:, self.robot["qvel_indices"]]
            power += (force*velocity).abs() * (~inactive)[:, None]
            contacts = self._contacts()
            for side in self.tip_force:
                self.tip_force[side][~inactive] = contacts[side][~inactive]
            for key, value in saved.items():
                self.views[key][inactive] = value
        self.tick.copy_(next_tick)
        power /= steps[:, None]
        for side in self.contact_history:
            self.contact_history[side] = torch.cat((self.contact_history[side][:, 1:],
                (self.tip_force[side].norm(dim=-1) > 0)[:, None]), dim=1)
        state, target = self.state(), self.target()
        reward, failure, details = self.reward(state, target, power, actions)
        if not torch.isfinite(reward).all() or not torch.isfinite(self.views["qpos"]).all():
            raise RuntimeError("Non-finite MJWarp rollout")
        self.done = failure | (self.tick == self.task["end_tick"])
        self.success_buf = ((self.tick == self.task["end_tick"]) & ~failure).float()
        self.failure_buf = failure.float()
        obs = self.features(state, self.target(self.task["control_ticks"]), self.sides, self.stage)
        # Task endpoint is a true finite-horizon terminal, not a time-limit truncation.
        return obs, reward, self.done, {"reward_dict": details, "error_masks": torch.ones_like(reward),
                                       "time_outs": torch.zeros_like(self.done)}

    def reward(self, state, target, power, actions):
        rewards, failures, details = [], [], {}
        for side in self.sides:
            hand = self.robot["hands"][side]
            ids = hand["indices"]
            bodies = torch.stack([state["bodies"][b] for b in hand["bodies"]], 1).clone()
            bodies[..., 3:7] = xyzw(bodies[..., 3:7])
            wrist = target[f"{side}_wrist"]
            actual = dict(base_state=bodies[:, 0], joints_state=bodies, q=state["robot_q"][:, ids[6:]],
                          dq=state["robot_dq"][:, ids[6:]])
            goal = dict(wrist_pos=wrist[:, :3], wrist_quat=xyzw(wrist[:, 3:7]), wrist_vel=wrist[:, 7:10],
                        wrist_ang_vel=wrist[:, 10:13], joints_pos=target[f"{side}_human"][:, 1:],
                        joints_vel=target[f"{side}_human_velocity"][:, 1:],
                        power=power[:, ids[6:]].sum(-1), wrist_power=power[:, ids[:6]].sum(-1))
            function = imitator_reward
            if self.stage == "residual":
                function = manipulation_reward
                part = self.metadata["part_names"].index(hand["object_part"])
                for values, source in ((actual, state["objects"][:, part]), (goal, target["object_state"][:, part])):
                    values.update(manip_obj_pos=source[:, :3], manip_obj_quat=xyzw(source[:, 3:7]),
                                  manip_obj_vel=source[:, 7:10], manip_obj_ang_vel=source[:, 10:13])
                goal.update(tip_force=state["tip_force"][side], tips_distance=target[f"{side}_tip_distance"],
                            tip_contact_state=self.contact_history[side])
            # Native failure grace periods are expressed at 60 Hz, regardless of task control rate.
            progress = self.tick*self.task["dt_s"]*60
            result = function(torch.zeros_like(self.tick), progress, progress, actions, actual, goal,
                              torch.full_like(progress, 1e9), 1.0, weight_indices())
            rewards.append(result[0])
            failures.append(result[3])
            details.update({f"{side}/{k}": v for k, v in result[4].items()})
        return torch.stack(rewards).sum(0), torch.stack(failures).any(0), details

    def set_train_info(self, frame, agent):
        pass

    def get_env_state(self):
        return None

    def set_env_state(self, state):
        if state is not None:
            raise ValueError("H2R does not restore simulator state from checkpoints")
