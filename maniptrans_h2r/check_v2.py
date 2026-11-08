"""GPU checks for v2 clock, reset, protocol features and nominal physics."""

from pathlib import Path

import numpy as np
import torch
import warp as wp

from .evaluate import Controller
from .io import read_json, write_json
from .observations import protocol_state
from .physics import gpu_audit
from .policy import Policy
from .residual_env import ResidualEnv


def check(args):
    root = Path(args.deployment)
    client = Controller(root, args.device)
    env = ResidualEnv(root / 'prepared', 2, args.device)
    task, robot = env.task, env.robot

    def initial():
        env.reset()
        env.views['qpos'][:] = torch.as_tensor(env.rsi['qpos'][0], device=args.device)
        env.views['qvel'].zero_()
        env.views['time'].zero_()
        env.views['qacc_warmstart'].zero_()
        env.tick.zero_()
        env.elapsed_ticks.zero_()
        env.views['ctrl'][:, robot['actuator_indices']] = client.initial[:, robot['ctrl_joint_indices']]
        env.sim.forward()
        env.tip_force = env.contacts()
        env.control.reset([0, 1], client.initial.expand(2, -1))

    initial()
    hello = dict(type='hello', protocol='h2r-evaluation-v1', **{k: task[k] for k in
        ('task_id', 'bundle_id', 'robot_manifest_sha256', 'robot_name', 'dt_s', 'control_ticks', 'end_tick', 'object', 'observation')},
        joint_names=robot['joint_names'], joint_units=task['semantics']['joint_units'],
        episodes=[dict(episode_id=f'e{i}', controller_seed=i+19) for i in range(2)])
    client.start(hello)
    policy = Policy(client.model, client.manifest['action_dim'], {'e0': 19, 'e1': 20}, args.device)
    state = env.state()
    frames = []
    for i in range(2):
        contacts = [dict(body1=body, body2=task['object']['parts'][hand['object_part']],
                         force_on_body2_world=(-state['tip_force'][side][i, tip]).tolist())
                    for side, hand in robot['hands'].items() for tip, body in enumerate(hand['contact_bodies'])]
        frames.append(dict(episode_id=f'e{i}', tick=0, done=False,
            **{k: state[k][i].tolist() for k in ('robot_q', 'robot_dq', 'object_q', 'object_dq')},
            bodies={k: v[i].tolist() for k, v in state['bodies'].items()}, contacts=dict(solve_tick=None, records=contacts)))
    wire = protocol_state(frames, robot, task, args.device)
    obs = env.features(state, env.target(1), env.sides, env.stage)
    for key, value in env.features(wire, env.target(1), env.sides, env.stage).items():
        torch.testing.assert_close(obs[key], value, rtol=0, atol=1e-6)
    commands = client.step(dict(type='frames', done=False, episodes=frames))['commands']
    actions = policy(obs, ['e0', 'e1'])
    targets = env.control.decode(actions, env.target(1)['robot_q'], env.sides, True)
    expected = torch.tensor([[commands[f'e{i}']['joint_targets'][n] for n in robot['joint_names']] for i in range(2)], device=args.device)
    torch.testing.assert_close(targets, expected, rtol=1e-5, atol=1e-6)
    env.step(actions)
    kept = {k: v[0].clone() for k, v in env.views.items()}
    history = {s: v[0].clone() for s, v in env.contact_history.items()}
    forces = {s: v[0].clone() for s, v in env.tip_force.items()}
    previous = env.control.previous[0].clone()
    env.done[:] = False
    env.done[1] = True
    env.reset_done()
    for key, value in kept.items():
        torch.testing.assert_close(env.views[key][0], value, rtol=0, atol=0)
    for side in env.sides:
        torch.testing.assert_close(env.contact_history[side][0], history[side], rtol=0, atol=0)
        torch.testing.assert_close(env.tip_force[side][0], forces[side], rtol=0, atol=0)
    torch.testing.assert_close(env.control.previous[0], previous, rtol=0, atol=0)
    initial()
    start_steps = env.common_steps
    for _ in range(60):
        env.done.zero_()  # Clock diagnostic only: allow native tracking failures.
        env.step(torch.zeros_like(actions))
    assert env.tick.tolist() == [120, 120] and env.common_steps-start_steps == 60
    torch.testing.assert_close(env.views['time'], torch.ones_like(env.views['time']), rtol=0, atol=2e-6)
    env.done.zero_()
    env.tick[:] = task['end_tick']-1
    before = env.views['time'].clone()
    _, _, done, info = env.step(torch.zeros_like(actions))
    assert done.all() and env.tick.tolist() == [task['end_tick']]*2 and not info['time_outs'].any()
    torch.testing.assert_close(env.views['time']-before, torch.full_like(before, 1/120), rtol=0, atol=1e-7)
    audit = gpu_audit(env.cpu, env.model, task, env.metadata['scene_sha256'])
    assert audit['fields'] == env.audit['fields'] and audit['options'] == env.audit['options']
    report = dict(task_id=task['task_id'], feature_and_target_parity=True, masked_reset=True,
                  sixty_controls_120_ticks=True, one_tick_terminal=True, nominal_physics_unchanged=True)
    if args.run:
        import json
        run = Path(args.run)
        manifest = read_json(run / 'run.json')
        if manifest['task_id'] != task['task_id'] or manifest['mode'] != 'sim':
            raise ValueError('Replay requires the same nominal task')
        rows = [json.loads(line) for line in (run / f"seed-{manifest['seeds'][0]}/physical.jsonl").read_text().splitlines()]
        initial()
        errors = []
        for row in rows[1:3]:
            target = torch.tensor(row['effective_target'], device=args.device)
            env.views['ctrl'][:, robot['actuator_indices']] = target[robot['ctrl_joint_indices']]
            env.sim.step()
            errors.append(dict(q=float(np.max(np.abs(env.views['qpos'][0, robot['qpos_indices']].cpu().numpy()-row['robot_q']))),
                force=float(np.max(np.abs(wp.to_torch(env.data.actuator_force)[0, robot['actuator_indices']].cpu().numpy()-np.asarray(row['actuator_force'])[robot['ctrl_joint_indices']])))))
        report['bench_first_interval'] = dict(backend=manifest['backend'], errors=errors)
        if not errors or max(e['q'] for e in errors) > 2e-5 or max(e['force'] for e in errors) > .001:
            raise AssertionError(report['bench_first_interval'])
    write_json(args.output, report)
    print(f'Passed mjlab GPU contracts: {args.output}')
