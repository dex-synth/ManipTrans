"""Persistent feedback controller for bench-owned physical evaluation (protocol h2r-evaluation-v2)."""

import argparse
import socket
import time

import torch

from .control import PositionControl
from .h2r_io import client
from .io import load_artifact, load_arrays, read_json, sha256, write_json
from .observations import Features, Reference, observed_bodies, protocol_state
from .policy import Policy, build, resolve_bases

FULL_CONTACTS = "robot-all-v1"


class Controller:
    def __init__(self, deployment, device):
        self.root, self.manifest = load_artifact(deployment, "deployment")
        prepared, meta = load_artifact(self.root / "prepared", "prepared")
        if meta["id"] != self.manifest["prepared_id"]:
            raise ValueError("Deployment reference binding differs")
        self.meta, self.device = meta, device
        self.robot = read_json(prepared / "robot_mapping.json")
        self.reference = Reference(load_arrays(prepared / "reference.npz"), device)
        initialization = read_json(prepared / "initialization.json")
        if initialization["joint_names"] != self.robot["joint_names"]:
            raise ValueError("Initial joint order differs")
        self.initial = torch.tensor([initialization["robot_q"]], dtype=torch.float32, device=device)
        self.features = Features(self.robot, meta["part_names"], meta["gravity_m_s2"])
        params = resolve_bases(self.manifest["params"], self.root)
        self.model = build(params, self.manifest["dimensions"], self.manifest["action_dim"], device)
        checkpoint = torch.load(self.root / self.manifest["checkpoint"], map_location=device, weights_only=True)
        self.model.load_state_dict(checkpoint["model"])
        self.control_dt_s = self.manifest["control_dt_s"]  # The policy's own period, as trained.
        self.controls, self.next_ticks = {}, {}

    def start(self, hello):
        if hello.get("type") != "hello" or hello.get("protocol") != client.PROTOCOL:
            raise ValueError(f"Unsupported evaluation protocol {hello.get('protocol')!r}; this client speaks {client.PROTOCOL}")
        if hello["task_id"] != self.manifest["task_id"]:
            raise ValueError("The evaluator runs a different task than this deployment was prepared for")
        observation = hello["observation"]
        if not observation["contacts"] or observation.get("contact_coverage") != FULL_CONTACTS:
            # The policy was trained on net fingertip forces from every partner, the robot included.
            raise ValueError(f"ManipTrans needs {FULL_CONTACTS} contact observations, the evaluation announces "
                             f"{observation.get('contact_coverage')!r}. Start evaluate-server with "
                             "contact_sensing.include_robot_robot=true")
        # The evaluation step is the evaluator's; only the control period is carried by the policy.
        self.ticks, self.dt_s, self.end_tick = client.control_ticks(hello, self.control_dt_s), hello["dt_s"], hello["end_tick"]
        self.orders = dict(robot=[hello["joint_names"].index(n) for n in self.robot["joint_names"]],
                           object=[list(hello["object"]["joints"]).index(n) for n in self.meta["object_joint_names"]])
        seeds = {e["episode_id"]: e["controller_seed"] for e in hello["episodes"]}
        if len(seeds) != len(hello["episodes"]):
            raise ValueError("Duplicate episode identity")
        self.policy = Policy(self.model, self.manifest["action_dim"], seeds, self.device, self.manifest["residual_mean"])
        for identity in seeds:
            control = PositionControl(self.robot, 1, self.device, self.manifest["control"])
            control.reset([0], self.initial)
            self.controls[identity] = control
            self.next_ticks[identity] = 0
        bodies = observed_bodies(self.robot, self.meta["object_bodies"])
        if not set(bodies) <= set(observation["bodies"]):
            raise ValueError("Required body observations unavailable")
        checkpoint = self.root / self.manifest["checkpoint"]
        return client.start_request(
            hello, method=self.manifest["method"], control_dt_s=self.control_dt_s, bodies=bodies, contacts=True,
            controller=dict(name=self.manifest["controller"], deployment_id=self.manifest["id"],
                            checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                            **self.manifest["training"]),
            initial_robot={e: dict(zip(self.robot["joint_names"], self.initial[0].tolist())) for e in seeds})

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
            if tick != self.next_ticks.get(frame["episode_id"]) or tick >= self.end_tick:
                raise ValueError("Unexpected control tick")
            if frame.get("contacts") is None or frame["contacts"]["solve_tick"] != (tick-1 if tick else None):
                raise ValueError("Missing or mistimed contacts")
        # As in training: observe now, track the reference one control period ahead (clamped to the end).
        ticks = torch.tensor([f["tick"] for f in active], device=self.device)
        target = self.reference.sample((ticks+self.ticks).clamp(max=self.end_tick)*self.dt_s)
        state = protocol_state(active, self.robot, self.meta["object_bodies"], self.orders, self.device)
        obs = self.features(state, target, self.robot["sides"], "residual")
        actions = self.policy(obs, identities)
        commands = {}
        for row, identity in enumerate(identities):
            positions = self.controls[identity].decode(actions[row:row+1], self.robot["sides"], True,
                                                       state["robot_q"][row:row+1])[0]
            commands[identity] = dict(tick=active[row]["tick"], joint_targets=dict(zip(self.robot["joint_names"], positions.tolist())))
            self.next_ticks[identity] = min(active[row]["tick"]+self.ticks, self.end_tick)
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
        client.send(stream, controller.start(client.receive(stream)))
        while True:
            frames = client.receive(stream)
            command = controller.step(frames)
            if command is None:
                break
            client.send(stream, command)
    for episode in frames["episodes"]:
        print(f"episode {episode['episode_id']} (seed {episode['seed']}): {episode.get('reason')} at tick {episode['tick']}")
    if args.diagnostics:
        write_json(args.diagnostics, dict(deployment_id=controller.manifest["id"],
                   elapsed_s=time.monotonic()-started, episodes=frames["episodes"]))


if __name__ == "__main__":
    main()
