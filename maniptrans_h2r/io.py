"""Content-checked method artifacts. Tasks are read by the vendored h2r_io loader."""

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


# Manifest file and format of each artifact. Earlier versions were built from the former task
# format, protocol and inherited training step; they are rejected, not converted.
ARTIFACTS = dict(pretrain=("prepared.json", "maniptrans-h2r-pretrain-v2"),
                 imitator=("imitator.json", "maniptrans-h2r-imitator-v2"),
                 prepared=("prepared.json", "maniptrans-h2r-prepared-v3"),
                 deployment=("deployment.json", "maniptrans-h2r-deployment-v3"))


def seal(root, kind, metadata):
    root = Path(root)
    name, format = ARTIFACTS[kind]
    result = dict(format=format, **metadata)
    result["files"] = {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*"))
                       if p.is_file() and p != root / name}
    result["id"] = digest(result)
    write_json(root / name, result)
    return result


def load_artifact(root, kind):
    root = Path(root).resolve()
    name, format = ARTIFACTS[kind]
    metadata = read_json(root / name)
    if metadata.get("format") != format:
        raise ValueError(f"{root.name}/{name} has format {metadata.get('format')!r}, this version reads {format!r}. "
                         "Artifacts made for an earlier task format cannot be reused: prepare and train again")
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
