"""Opt-in, bounded episode truncation for local MJWarp EPA failures."""

from collections import deque
import json
from pathlib import Path

import torch
import warp as wp

from .io import write_json


EPA_RECOVERY = dict(schema="epa-episode-skip-v1", flags=["EPA_HORIZON"],
                    max_world_fraction=.001, window_controls=1000, max_skipped_episodes=32,
                    reset_overflow="fatal")
STATE_KEYS = ("qpos", "qvel", "act", "ctrl", "time", "qacc_warmstart")


class EpaRecovery:
    def __init__(self, output):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.events = 0
        self.skipped_episodes = 0
        self.recent = deque()
        write_json(self.output / "summary.json", self.state_dict())

    def state_dict(self):
        return dict(events=self.events, skipped_episodes=self.skipped_episodes, recent=list(self.recent))

    def load_state_dict(self, state):
        self.events = state["events"]
        self.skipped_episodes = state["skipped_episodes"]
        self.recent = deque(state["recent"])
        write_json(self.output / "summary.json", self.state_dict())

    def handle(self, env, error, before, invalid, substep):
        if error.report["phase"] != "step" or set(error.report["flags"]) != {"EPA_HORIZON"}:
            raise error
        if error.world_ids is None:
            raise error
        if any(not torch.isfinite(env.views[k]).all() for k in ("qpos", "qvel")):
            raise FloatingPointError("Non-finite physics state during EPA recovery")
        ids = error.world_ids.to(env.device)
        new_count = int((~invalid[ids]).sum())
        step = env.common_steps
        while self.recent and self.recent[0][0] <= step-EPA_RECOVERY["window_controls"]:
            self.recent.popleft()
        reason = None
        if int(invalid.sum())+new_count > max(1, int(env.count*EPA_RECOVERY["max_world_fraction"])):
            reason = "too_many_affected_worlds"
        elif sum(n for _, n in self.recent)+new_count > EPA_RECOVERY["max_skipped_episodes"]:
            reason = "too_many_skipped_episodes"
        self.events += 1
        indices = error.world_ids.tolist()
        snapshot = f"event-{self.events:06d}.pth"
        torch.save({k: before[k][ids].detach().cpu() for k in STATE_KEYS}, self.output / snapshot)
        references = [dict(world=i, window=int(env.motion.window_ids[i]),
                           sample=int(env.motion.positions[i]),
                           reference=env.motion.windows[env.motion.window_ids[i]]) for i in indices]
        record = dict(event=self.events, common_control_step=step, substep=substep,
                      status=reason or "episode_skipped", new_skipped_episodes=new_count,
                      overflow=error.report, worlds=indices, references=references,
                      pre_substep_state=snapshot)
        with (self.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record)+"\n")
        if reason:
            error.report["recovery_refused"] = reason
            raise error
        self.skipped_episodes += new_count
        if new_count:
            self.recent.append((step, new_count))
        write_json(self.output / "summary.json", self.state_dict())
        print(f"EPA recovery: control={step}, worlds={indices}, skipped={self.skipped_episodes}", flush=True)
        # MJWarp overflow bits persist until reset. Clear only the handled local bits.
        wp.to_torch(env.data.overflow)[ids] = 0
        return ids
