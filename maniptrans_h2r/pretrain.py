"""Prepare, train and diagnose a generalist single-hand ManipTrans imitator."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil

import numpy as np

from .clock import CONTROL_DT_S, PHYSICS_DT_S
from .h2r_io import load_task
from .io import check_files, digest, read_json, seal, sha256, write_json
from .motion import WIDTH, WINDOW_SCHEMA
from .pretrain_worker import ordered_results, process, worker_pool
from .robot import command_limits, mapping
from .robot_physics import extract_hand

CLIP_FILTER = dict(min_source_samples=4, min_duration_s=.5, duration_tolerance_s=1e-10)


def select_clips(clips):
    """Apply the imitator's minimum length before interpolation or synthesis."""
    selected, excluded = [], []
    for clip in clips:
        if (clip["num_samples"] < CLIP_FILTER["min_source_samples"] or
                clip["duration_s"] < CLIP_FILTER["min_duration_s"]-CLIP_FILTER["duration_tolerance_s"]):
            excluded.append(dict(clip_id=clip["id"], dataset=clip["dataset"], recording=clip["recording"],
                source_side=clip["source_side"], source_interval=clip["source_interval"],
                num_samples=clip["num_samples"], duration_s=clip["duration_s"], reason="too_short"))
        else:
            selected.append(clip)
    return selected, excluded


def prepare(args):
    import mujoco
    import torch
    torch.set_num_threads(4)
    workers = getattr(args, "workers", 8)
    if workers < 1:
        raise ValueError("--workers must be positive")
    corpus = Path(args.corpus).resolve()
    meta = read_json(corpus / "corpus.json")
    if meta["format"] != "h2r-hand-motion-corpus-v2" or digest({k: v for k, v in meta.items() if k != "id"}) != meta["id"]:
        raise ValueError("Unsupported or changed hand corpus; export it with the current h2r-bench export-hand-motions "
                         "(h2r-hand-motion-corpus-v2, the common MANO convention)")
    check_files(corpus, meta["files"])
    if not meta["inventory_complete"] and not args.allow_incomplete:
        raise ValueError("Incomplete source corpus; use --allow-incomplete only for checks")
    clips = [json.loads(line) for line in (corpus / "clips.jsonl").read_text().splitlines() if line.strip()]
    task = load_task(args.task)
    sides = ("right", "left") if args.side == "both" else (args.side,)
    if not set(sides) <= set(task.hand_sides):
        raise ValueError(f"--side {args.side} needs a task whose robot has those hands; this one has {list(task.hand_sides)}")
    output = Path(args.output).resolve()
    if output.is_relative_to(task.path) or output.is_relative_to(corpus):
        raise ValueError("Output must be outside its sealed inputs")
    output.mkdir(parents=True, exist_ok=False)
    clips, excluded = select_clips(clips)
    write_json(output / "clip_selection.json", dict(rule=CLIP_FILTER, selected_clip_count=len(clips), excluded_clips=excluded))
    if not clips:
        raise ValueError("No clips meet the imitator minimum length; see clip_selection.json")
    mano_hashes = {side: sha256(Path(args.mano_root) / f"MANO_{side.upper()}.pkl") for side in {c["source_side"] for c in clips}}
    if any(mano_hashes[c["source_side"]] != c["mano_sha256"] for c in clips):
        raise ValueError("MANO assets differ from exported corpus")
    outputs = {side: output / side if args.side == "both" else output for side in sides}
    model = mujoco.MjModel.from_xml_path(str(task.scene_xml))
    robot = mapping(model, task)
    states = {}
    for side, destination in outputs.items():
        hand_model, hand_robot, compatibility = extract_hand(task, robot, side, destination)
        command_limits(hand_model, hand_robot)
        write_json(destination / "robot_mapping.json", hand_robot)
        if destination != output:
            shutil.copy2(output / "clip_selection.json", destination / "clip_selection.json")
        (destination / ".parts").mkdir()
        states[side] = dict(records=[], errors=[], samples=0, compatibility=compatibility)
    workers = min(workers, len(clips))
    initargs = (corpus, outputs, args.mano_root, args.seed, args.synthetic_fraction, args.allow_incomplete)
    try:
        with ExitStack() as stack:
            streams = {side: stack.enter_context((path / "motion.raw").open("wb")) for side, path in outputs.items()}
            executor = stack.enter_context(worker_pool(workers, initargs))
            results = ordered_results(executor, process, enumerate(clips), 2*workers)
            for clip, result in zip(clips, results):
                for side, row in result.items():
                    state, destination = states[side], outputs[side]
                    state["errors"].extend(row["errors"])
                    if row["errors"] and not args.allow_incomplete:
                        write_json(destination / "reference_errors.json", state["errors"])
                        raise ValueError(row["errors"][0]["reason"])
                    with Path(row["path"]).open("rb") as part:
                        shutil.copyfileobj(part, streams[side], length=1024*1024)
                    Path(row["path"]).unlink()
                    for record in row["records"]:
                        record["offset"] += state["samples"]
                    state["records"].extend(row["records"])
                    state["samples"] += row["samples"]
                    print(f"Prepared {side} {clip['dataset']}/{clip['id'][:12]}: {state['samples']} samples", flush=True)
    finally:
        for destination in outputs.values():
            shutil.rmtree(destination / ".parts")
    for side, destination in outputs.items():
        finish_preparation(destination, side, states[side], meta, task, args)
    return output


def finish_preparation(output, side, state, meta, task, args):
    records, errors, offset = state["records"], state["errors"], state["samples"]
    raw = output / "motion.raw"
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
        seed=args.seed, corpus_id=meta["id"], augmentation="interpolated_mano_v1", samples=offset,
        windowing=WINDOW_SCHEMA, short_window_count=sum(r["length"] < 31 for r in records)))
    diagnostic = []
    for dataset in ("arctic", "oakink2", "hot3d", "taco"):
        for source_side in ("left", "right"):
            choices = sorted((i for i, r in enumerate(records) if not r["synthetic"] and r["dataset"] == dataset and r["source_side"] == source_side),
                             key=lambda i: digest(records[i]))
            indices = np.linspace(0, len(choices)-1, min(32, len(choices)), dtype=int)
            diagnostic.extend(choices[i] for i in indices)
    write_json(output / "diagnostic_manifest.json", dict(windows=diagnostic, overlap="training-corpus tracking diagnostic"))
    result = seal(output, "pretrain", dict(side=side,
        corpus_id=meta["id"], partial=bool(not meta["inventory_complete"] or errors),
        compatibility=state["compatibility"], control_dt_s=CONTROL_DT_S, physics_dt_s=PHYSICS_DT_S,
        source_task_id=task.task_id,
        recipe=dict(seed=args.seed, synthetic_fraction=args.synthetic_fraction, augmentation="interpolated_mano_v1",
                    windowing=WINDOW_SCHEMA, clip_filter=CLIP_FILTER)))
    print(f"Prepared generalist reference: {result['id']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--corpus", required=True)
    prep.add_argument("--task", required=True, help="Task directory whose robot provides the hand model")
    prep.add_argument("--side", choices=("left", "right", "both"), required=True,
                      help="both shares human decoding and writes OUTPUT/right and OUTPUT/left")
    prep.add_argument("--workers", type=int, default=8, help="Clip worker processes, each with one numerical thread (default: 8)")
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
