"""Run the fixed OakInk2 development batch from explicit, portable input paths.

Manifest: {bench_command: [...], bench_cwd, profiles: [{id, profile, bundle,
robot_bundle, task, spec, prepared, right_base, left_base, deployment}],
included_snapshot: path, included_sha256: hash}. Paths must exist in the process
namespace. Commands run sequentially; failures remain in results.json.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml


def execute(command, log, cwd=None):
    with log.open('w') as stream:
        try:
            code = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=False).returncode
        except OSError as exc:
            stream.write(str(exc)+'\n')
            code = 127
    return dict(command=command, returncode=code, log=str(log), status='passed' if code == 0 else 'failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--phase', choices=['prepare', 'smoke', 'pilot', 'train', 'eval'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    snapshot = Path(manifest['included_snapshot'])
    if hashlib.sha256(snapshot.read_bytes()).hexdigest() != manifest['included_sha256']:
        raise ValueError('Included snapshot changed')
    included = yaml.safe_load(snapshot.read_text())['included']
    profiles = manifest['profiles']
    if (len(profiles) != 10 or {p['profile'] for p in profiles} != set(included)
            or {p['id'] for p in profiles} != {f'O{i}' for i in range(1, 11)}):
        raise ValueError('Manifest must cover exactly the fixed ten included profiles')
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    def save():
        (args.output/'results.json').write_text(json.dumps(dict(phase=args.phase, manifest=manifest,
            scheduled_profiles=len(profiles), scheduled_episodes=80 if args.phase == 'eval' else None,
            results=results), indent=2)+'\n')
    save()
    bench = manifest['bench_command']
    bench_cwd = manifest['bench_cwd']
    method = [sys.executable, '-m']
    subset = {'smoke': {'O4', 'O2', 'O6', 'O9', 'O10'}, 'pilot': {'O4', 'O2', 'O9', 'O10'}}
    for row in profiles:
        if row['id'] not in subset.get(args.phase, {p['id'] for p in profiles}):
            continue
        name = row['id']
        report = dict(id=name, profile=row['profile'], steps=[])
        results.append(report)
        save()
        if args.phase == 'prepare':
            commands = [
                (bench+['prepare-evaluation', '--scene-bundle', row['bundle'], '--robot-bundle', row['robot_bundle'],
                        '--physics-hz', '120', '--control-hz', '60', '--contacts', '--output', row['task']], bench_cwd),
                (bench+['prepare-metrics', f"task={row['task']}", f"output={row['spec']}", 'metrics.eval_hz=120'], bench_cwd),
                (method+['maniptrans_h2r.prepare', '--task', row['task'], '--output', row['prepared'], '--iterations', '4000', '--device', args.device], None)]
        elif args.phase in {'smoke', 'pilot', 'train'}:
            epochs = {'smoke': 2, 'pilot': 200, 'train': 1000}[args.phase]
            destination = row['deployment'] if args.phase == 'train' else row['deployment']+'-'+args.phase
            command = method+['maniptrans_h2r.train', '--prepared', row['prepared'], '--imitator-right', row['right_base'],
                '--imitator-left', row['left_base'], '--output', destination, '--epochs', str(epochs),
                '--worlds', '8' if args.phase == 'smoke' else '4096', '--seed', '42', '--device', args.device]
            if args.phase == 'smoke':
                command += ['--small-network', '--allow-smoke-imitator']
            if args.phase == 'train' and name in subset['pilot']:
                command += ['--resume', row['deployment']+'-pilot/resume.pth']
            commands = [(command, None)]
        else:
            # Evaluation commands are explicit so callers can use bench batch paths
            # and either CPU or GPU backend installations without importing bench.
            scheduled = row.get('evaluation', [])
            expected = {(b, m, s) for b in ('mujoco', 'mjwarp') for m in ('sim', 'sim-dr') for s in (11, 23)}
            if {(e['backend'], e['mode'], e['seed']) for e in scheduled} != expected or len(scheduled) != 8:
                raise ValueError(f'{name}: declare all eight evaluation episodes')
            for index, episode in enumerate(scheduled):
                server_log = args.output/f'{name}-eval-{index}-server.log'
                with server_log.open('w') as stream:
                    try:
                        server = subprocess.Popen(episode['server_command'], cwd=bench_cwd, stdout=stream, stderr=subprocess.STDOUT)
                    except OSError as exc:
                        report['status'] = 'failed'
                        report['steps'].append(dict(episode=episode, status='failed', reason=str(exc)))
                        save()
                        continue
                    try:
                        client = execute(method+['maniptrans_h2r.evaluate', '--deployment', row['deployment'],
                            '--socket', episode['socket'], '--device', 'cpu'], args.output/f'{name}-eval-{index}-client.log')
                        try:
                            code = server.wait(timeout=60)
                        except subprocess.TimeoutExpired:
                            server.terminate(); code = server.wait(timeout=30)
                        report['steps'].append(dict(episode=episode, client=client, server_returncode=code,
                            status='passed' if code == 0 and client['returncode'] == 0 else 'failed'))
                        if code or client['returncode']:
                            report['status'] = 'failed'
                        save()
                    finally:
                        if server.poll() is None:
                            server.terminate(); server.wait(timeout=30)
                if 'score_command' in episode:
                    score = execute(episode['score_command'], args.output/f'{name}-eval-{index}-score.log', bench_cwd)
                    report['steps'].append(score)
                    if score['returncode']:
                        report['status'] = 'failed'
                    save()
            commands = []
        for index, (command, cwd) in enumerate(commands):
            step = execute(command, args.output/f'{name}-{index}.log', cwd)
            report['steps'].append(step)
            if step['returncode']:
                report['status'] = 'failed'
                break
        report.setdefault('status', 'completed')
        save()
    print(args.output/'results.json')


if __name__ == '__main__':
    main()
