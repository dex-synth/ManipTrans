"""Portable, content-checked inputs. No simulator or dataset runtime imports."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np


def digest(value):
    data = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Non-finite JSON value: {value}")

    def number(value):
        result = float(value)
        return result if math.isfinite(result) else invalid(value)

    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid, parse_float=number)


def local_file(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"Dependency escapes artifact: {name}")
    return path


def check_files(root, files):
    for name, expected in files.items():
        if sha256(local_file(root, name)) != expected:
            raise ValueError(f"Changed dependency: {name}")


def load_task(root):
    root = Path(root).resolve()
    task = read_json(root / "task.json")
    identity = task.pop("task_id")
    if task["format"] not in {f"h2r-evaluation-v{i}" for i in (1, 2, 3)}:
        raise ValueError("Unsupported task format")
    if digest(task) != identity:
        raise ValueError("Changed task identity")
    task["task_id"] = identity
    check_files(root, task["files"])
    for required in (task["scene"], "reference.json", "reference.npz"):
        if required not in task["files"]:
            raise ValueError(f"Unsealed task dependency: {required}")
    if not task["observation"]["contacts"]:
        raise ValueError("ManipTrans requires a common task prepared with --contacts")
    if task["observation"].get("contact_schema") != "h2r-body-contacts-v1":
        raise ValueError("Unsupported contact observations")
    return root, task


def check_inputs(bundle, robot, task_path, task):
    bundle, robot = Path(bundle).resolve(), Path(robot).resolve()
    manifest = read_json(bundle / "bundle.json")
    if manifest.get("kind", "physical") != "physical":
        raise ValueError("A physical preprocessing bundle is required")
    if manifest["format"] not in {f"h2r-preprocess-v{i}" for i in (1, 2, 3)}:
        raise ValueError("Unsupported bundle format")
    check_files(bundle, manifest["files"])
    if digest(manifest["files"]) != manifest["bundle_id"] or manifest["bundle_id"] != task["bundle_id"]:
        raise ValueError("Bundle and task identities differ")
    if sha256(robot / "robot.json") != task["robot_manifest_sha256"]:
        raise ValueError("Robot and task identities differ")
    check_files(robot, read_json(robot / "robot.json")["files_sha256"])
    for name in ("reference.json", "reference.npz"):
        if sha256(bundle / name) != sha256(task_path / name):
            raise ValueError(f"Task reference differs from bundle: {name}")


def seal(root, name, metadata):
    root = Path(root)
    result = dict(metadata)
    result["files"] = {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*"))
                       if p.is_file() and p != root / name}
    result["id"] = digest(result)
    write_json(root / name, result)
    return result


def load_artifact(root, name="prepared.json"):
    root = Path(root).resolve()
    metadata = read_json(root / name)
    identity = metadata.pop("id")
    if digest(metadata) != identity:
        raise ValueError(f"Changed {name} identity")
    metadata["id"] = identity
    check_files(root, metadata["files"])
    return root, metadata


def load_arrays(path):
    with np.load(path, allow_pickle=False) as arrays:
        result = {name: arrays[name] for name in arrays.files}
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError("Non-finite reference array")
    return result
