"""Persistent feedback controller for bench-owned physical evaluation."""

import argparse
import socket
import time

import torch

from .control import PositionControl
from .io import load_artifact, load_arrays, load_task, read_json, write_json
from .observations import Features, Reference, protocol_state
from .policy import Policy, build, resolve_bases
from .protocol import receive, send


class Controller:
    def __init__(self, deployment, device):
        self.root, self.manifest = load_artifact(deployment, "deployment.json")
        if self.manifest["format"] not in {"maniptrans-h2r-deployment-v1", "maniptrans-h2r-deployment-v2"}:
            raise ValueError("Unsupported deployment format")
        prepared, meta = load_artifact(self.root / "prepared")
        if meta["id"] != self.manifest["prepared_id"]:
            raise ValueError("Deployment reference binding differs")
        _, self.task = load_task(prepared / "task")
        if self.manifest.get("recipe", {}).get("training_revision", 0) >= 4:
            from .robot_physics import require_contacts
            require_contacts(self.task)
        self.robot = read_json(prepared / "robot_mapping.json")
        self.reference = Reference(load_arrays(prepared / "reference.npz"), device)
        self.initial = self.reference.values["robot_q"][0:1]
        if self.manifest["format"] == "maniptrans-h2r-deployment-v2":
            initialization = read_json(prepared / "initialization.json")
            if initialization["joint_names"] != self.robot["joint_names"]:
                raise ValueError("Initial joint order differs")
            self.initial = torch.tensor([initialization["robot_q"]], dtype=torch.float32, device=device)
        self.features = Features(self.robot, meta["part_names"],
            sum(v*v for v in read_json(prepared / "task/scene/compiled.json")["base_scene"]["options"]["gravity"])**0.5)
        self.device = device
        params = resolve_bases(self.manifest["params"], self.root)
        self.model = build(params, self.manifest["dimensions"], self.manifest["action_dim"], device)
        checkpoint = torch.load(self.root / self.manifest["checkpoint"], map_location=device, weights_only=True)
        self.model.load_state_dict(checkpoint["model"])
        self.controls, self.next_ticks = {}, {}

    def start(self, hello):
        if hello.get("type") != "hello" or hello.get("protocol") != "h2r-evaluation-v1":
            raise ValueError("Unsupported evaluation protocol")
        for key in ("task_id", "bundle_id", "robot_manifest_sha256", "robot_name", "dt_s", "control_ticks", "end_tick", "object", "observation"):
            if hello.get(key) != self.task[key]:
                raise ValueError(f"Evaluation task differs: {key}")
        if hello["joint_names"] != self.robot["joint_names"] or hello["joint_units"] != self.task["semantics"]["joint_units"]:
            raise ValueError("Evaluation joint names/units differ")
        seeds = {e["episode_id"]: e["controller_seed"] for e in hello["episodes"]}
        if len(seeds) != len(hello["episodes"]):
            raise ValueError("Duplicate episode identity")
        self.policy = Policy(self.model, self.manifest["action_dim"], seeds, self.device, self.manifest["residual_mean"])
        initial = self.initial
        for identity in seeds:
            control = PositionControl(self.robot, 1, self.device, self.manifest["control"])
            control.reset([0], initial)
            self.controls[identity] = control
            self.next_ticks[identity] = 0
        bodies = list(dict.fromkeys([b for h in self.robot["hands"].values() for b in h["bodies"]]
                                    + list(self.task["object"]["parts"].values())))
        if not set(bodies) <= set(hello["observation"]["bodies"]):
            raise ValueError("Required body observations unavailable")
        return dict(type="start", task_id=self.task["task_id"],
            binding={"bundle_id": self.task["bundle_id"], "robot_manifest_sha256": self.task["robot_manifest_sha256"],
                     "method": self.manifest["method"], "controller": self.manifest["id"]},
            requirements=dict(bodies=bodies, contacts=True, control_dt_s=self.task["control_ticks"]*self.task["dt_s"]),
            initial_robot={e: dict(zip(self.robot["joint_names"], initial[0].tolist())) for e in seeds})

    @torch.no_grad()
    def step(self, message):
        if message.get("type") == "error":
            raise RuntimeError(f"Evaluation server rejected controller: {message}")
        if message.get("type") != "frames":
            raise ValueError("Expected evaluation frames")
        active = [f for f in message["episodes"] if not f["done"]]
        if message["done"] != (not active):
            raise ValueError("Inconsistent episode lifecycle")
        if not active:
            return None
        identities = [f["episode_id"] for f in active]
        if len(set(identities)) != len(identities):
            raise ValueError("Duplicate episode frame")
        for frame in active:
            tick = frame["tick"]
            if tick != self.next_ticks.get(frame["episode_id"]) or tick >= self.task["end_tick"]:
                raise ValueError("Unexpected control tick")
            if frame.get("contacts") is None or frame["contacts"]["solve_tick"] != (tick-1 if tick else None):
                raise ValueError("Missing or mistimed contacts")
        ticks = torch.tensor([f["tick"] for f in active], device=self.device)
        target = self.reference.sample((ticks+self.task["control_ticks"]).clamp(max=self.task["end_tick"])*self.task["dt_s"])
        state = protocol_state(active, self.robot, self.task, self.device)
        obs = self.features(state, target, self.robot["sides"], "residual")
        actions = self.policy(obs, identities)
        commands = {}
        for row, identity in enumerate(identities):
            positions = self.controls[identity].decode(actions[row:row+1], target["robot_q"][row:row+1],
                                                       self.robot["sides"], True)[0]
            commands[identity] = dict(tick=active[row]["tick"], joint_targets=dict(zip(self.robot["joint_names"], positions.tolist())))
            self.next_ticks[identity] = min(active[row]["tick"]+self.task["control_ticks"], self.task["end_tick"])
        return dict(type="step", commands=commands)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timeout-s", type=float, default=120)
    parser.add_argument("--diagnostics")
    args = parser.parse_args()
    torch.set_num_threads(4)
    controller = Controller(args.deployment, args.device)
    started = time.monotonic()
    with socket.socket(socket.AF_UNIX) as stream:
        stream.settimeout(args.timeout_s)
        while True:
            try:
                stream.connect(args.socket)
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic()-started >= args.timeout_s:
                    raise TimeoutError("Evaluation server did not start")
                time.sleep(0.2)
        send(stream, controller.start(receive(stream)))
        while True:
            frames = receive(stream)
            command = controller.step(frames)
            if command is None:
                break
            send(stream, command)
    if args.diagnostics:
        write_json(args.diagnostics, dict(deployment_id=controller.manifest["id"],
                   elapsed_s=time.monotonic()-started, episodes=frames["episodes"]))


if __name__ == "__main__":
    main()
