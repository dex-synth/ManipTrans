"""Prepare, train and diagnose a generalist single-hand ManipTrans imitator."""

import argparse
import json
import math
from pathlib import Path
import shutil

import numpy as np

from .hand_reference import interpolate_motion, reference_from_points
from .io import check_files, digest, load_arrays, load_artifact, load_task, read_json, seal, sha256, write_json
from .motion import WIDTH, pack, windows
from .robot import command_limits, mapping
from .robot_physics import extract_hand


def prepare(args):
    import mujoco
    import torch
    torch.set_num_threads(4)
    corpus = Path(args.corpus).resolve()
    meta = read_json(corpus / "corpus.json")
    if meta["format"] != "h2r-hand-motion-corpus-v1" or digest({k: v for k, v in meta.items() if k != "id"}) != meta["id"]:
        raise ValueError("Unsupported or changed hand corpus")
    check_files(corpus, meta["files"])
    if not meta["inventory_complete"] and not args.allow_incomplete:
        raise ValueError("Incomplete source corpus; use --allow-incomplete only for checks")
    clips = [json.loads(line) for line in (corpus / "clips.jsonl").read_text().splitlines() if line.strip()]
    source, task = load_task(args.task)
    output = Path(args.output).resolve()
    if output.is_relative_to(source) or output.is_relative_to(corpus):
        raise ValueError("Output must be outside its sealed inputs")
    output.mkdir(parents=True, exist_ok=False)
    model = mujoco.MjModel.from_xml_path(str(source / task["scene"]))
    robot = mapping(model, task)
    model, robot, compatibility = extract_hand(source, task, robot, args.side, output)
    command_limits(model, robot)
    write_json(output / "robot_mapping.json", robot)
    records, errors, offset = [], [], 0
    raw = output / "motion.raw"
    with raw.open("wb") as stream:
        for clip in clips:
            if sha256(Path(args.mano_root) / f"MANO_{clip['source_side'].upper()}.pkl") != clip["mano_sha256"]:
                raise ValueError("MANO assets differ from exported corpus")
            arrays = load_arrays(corpus / clip["path"])
            # Check exact source-pose reconstruction before enabling interpolation/augmentation.
            reconstructed = interpolate_motion(arrays, clip, arrays["time_s"], args.mano_root)
            if np.max(np.abs(reconstructed-arrays["joints_m"])) > 1e-5:
                raise ValueError(f"MANO reconstruction differs: {clip['id']}")
            duration = float(arrays["time_s"][-1])
            rng = np.random.default_rng(int(digest([args.seed, clip["id"]])[:16], 16))
            recipes = [(False, None, duration)]
            if args.synthetic_fraction:
                for _ in range(max(1, math.ceil(duration/10))):
                    start = rng.uniform(0, max(0, duration-.5))
                    length = rng.uniform(.5, min(5, duration-start))
                    recipes.append((True, [start, start+length], length))
            for synthetic, endpoints, length in recipes:
                times = np.arange(int(np.floor(length*60+1e-9))+1)/60
                try:
                    points = interpolate_motion(arrays, clip, times, args.mano_root,
                                                endpoints=endpoints, mirror=clip["source_side"] != args.side)
                    reference = reference_from_points(model, robot, args.side, points, times)
                    values = pack(reference, args.side)
                except ValueError as exc:
                    errors.append(dict(clip_id=clip["id"], dataset=clip["dataset"], source_interval=clip["source_interval"],
                                       synthetic=synthetic, endpoints=endpoints, reason=str(exc)))
                    if not args.allow_incomplete:
                        write_json(output / "reference_errors.json", errors)
                        raise
                    continue
                stream.write(values.tobytes())
                for start, count in windows(len(times)):
                    records.append(dict(offset=offset+start, length=count, dataset=clip["dataset"],
                        source_side=clip["source_side"], synthetic=synthetic, clip_id=clip["id"],
                        recording=clip["recording"], endpoints=endpoints, unused_tail_s=length-times[-1]))
                offset += len(times)
            print(f"Prepared {clip['dataset']}/{clip['id'][:12]}: {offset} samples", flush=True)
    if not offset:
        raise ValueError("No valid robot hand references")
    target = np.lib.format.open_memmap(output / "motion.npy", mode="w+", dtype=np.float32, shape=(offset, WIDTH))
    source_array = np.memmap(raw, mode="r", dtype=np.float32, shape=(offset, WIDTH))
    for start in range(0, offset, 65536):
        target[start:start+65536] = source_array[start:start+65536]
    target.flush()
    del source_array, target
    raw.unlink()
    write_json(output / "reference_errors.json", errors)
    write_json(output / "motion_manifest.json", dict(windows=records, synthetic_fraction=args.synthetic_fraction,
        seed=args.seed, corpus_id=meta["id"], augmentation="interpolated_mano_v1", samples=offset))
    diagnostic = []
    for dataset in ("arctic", "oakink2", "hot3d", "taco"):
        for side in ("left", "right"):
            choices = sorted((i for i, r in enumerate(records) if not r["synthetic"] and r["dataset"] == dataset and r["source_side"] == side),
                             key=lambda i: digest(records[i]))
            indices = np.linspace(0, len(choices)-1, min(32, len(choices)), dtype=int)
            diagnostic.extend(choices[i] for i in indices)
    write_json(output / "diagnostic_manifest.json", dict(windows=diagnostic, overlap="training-corpus tracking diagnostic"))
    result = seal(output, "prepared.json", dict(format="maniptrans-h2r-pretrain-v1", side=args.side,
        corpus_id=meta["id"], partial=bool(not meta["inventory_complete"] or errors),
        compatibility=compatibility, control_dt_s=1/60, physics_dt_s=1/120, source_task_id=task["task_id"],
        recipe=dict(seed=args.seed, synthetic_fraction=args.synthetic_fraction, augmentation="interpolated_mano_v1")))
    print(f"Prepared generalist reference: {result['id']}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--corpus", required=True)
    prep.add_argument("--task", required=True)
    prep.add_argument("--side", choices=("left", "right"), required=True)
    prep.add_argument("--mano-root", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--synthetic-fraction", type=float, default=.1)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--allow-incomplete", action="store_true")
    training = sub.add_parser("train")
    from .training import training_arguments
    training_arguments(training, "imitator")
    check = sub.add_parser("check")
    check.add_argument("--imitator", required=True)
    check.add_argument("--prepared", required=True)
    check.add_argument("--output", required=True)
    check.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "prepare":
        if not 0 <= args.synthetic_fraction < 1:
            parser.error("--synthetic-fraction must be in [0, 1)")
        prepare(args)
    elif args.command == "train":
        from .training import train_imitator
        train_imitator(args)
    else:
        from .diagnostic import check_imitator
        check_imitator(args)


if __name__ == "__main__":
    main()
