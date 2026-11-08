"""A failed controller/server must remain in the fixed batch denominator."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import yaml


def test_batch_keeps_all_failed_evaluation_episodes(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2]/'scripts/h2r_oakink2_batch.py'
    spec = importlib.util.spec_from_file_location('oakink_batch', path)
    batch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(batch)
    snapshot = tmp_path/'included.yaml'
    snapshot.write_text(yaml.safe_dump(dict(included=[f'profile-{i}' for i in range(1, 11)])))
    episodes = [dict(backend=b, mode=m, seed=s, server_command=['server'], socket='socket')
                for b in ('mujoco', 'mjwarp') for m in ('sim', 'sim-dr') for s in (11, 23)]
    manifest = tmp_path/'inputs.json'
    manifest.write_text(json.dumps(dict(included_snapshot=str(snapshot),
        included_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(), bench_command=['bench'], bench_cwd=str(tmp_path),
        profiles=[dict(id=f'O{i}', profile=f'profile-{i}', deployment='deployment', evaluation=episodes) for i in range(1, 11)])))
    monkeypatch.setattr(batch.subprocess, 'Popen', lambda *a, **k: SimpleNamespace(wait=lambda **kw: 1, poll=lambda: 1))
    monkeypatch.setattr(batch, 'execute', lambda *a, **k: dict(returncode=1, status='failed'))
    output = tmp_path/'output'
    monkeypatch.setattr(sys, 'argv', ['batch', '--manifest', str(manifest), '--phase', 'eval', '--output', str(output)])
    batch.main()
    report = json.loads((output/'results.json').read_text())
    assert report['scheduled_episodes'] == 80
    assert len(report['results']) == 10
    assert all(row['status'] == 'failed' and len(row['steps']) == 8 for row in report['results'])
    assert all(step['status'] == 'failed' for row in report['results'] for step in row['steps'])
