"""Prepare and train a residual for every case of a suite, in distribution order.

    python -m maniptrans_h2r.suite --suite SUITE --output OUTPUT --imitator BASE [BASE ...] [-- TRAIN OPTIONS]

Each case gets OUTPUT/<name>/prepared and OUTPUT/<name>/deployment, where <name> is the case_id
with "/" replaced by "-" and "+" by "_" (the name `h2r-bench declare-evaluation --suite` uses).
The base of each hand is chosen among the given ones by side and hand-physics signature.
Finished steps are skipped; every case records its outcome in OUTPUT/<name>/result.json, a failed
case does not stop later ones, and several runs with disjoint --case sets may share one OUTPUT.
"""

import argparse
from pathlib import Path
import subprocess
import sys

from .h2r_io import load_suite
from .io import load_artifact, write_json
from .robot_physics import same_signature


def run(module, arguments, log):
    with log.open("w") as stream:
        return subprocess.run([sys.executable, "-m", f"maniptrans_h2r.{module}", *arguments],
                              stdout=stream, stderr=subprocess.STDOUT, check=False).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--imitator", nargs="+", required=True, help="Generalist base directories")
    parser.add_argument("--case", action="append", help="Only this case_id (repeatable; default: every case)")
    parser.add_argument("--iterations", type=int, default=4000, help="IK iteration limit of each preparation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("train", nargs=argparse.REMAINDER, help="After --: options passed to maniptrans_h2r.train")
    args = parser.parse_args()
    train_options = args.train[1:] if args.train[:1] == ["--"] else args.train  # argparse keeps the separator before 3.13.
    cases = load_suite(args.suite).cases
    unknown = set(args.case or ()) - {case.case_id for case in cases}
    if unknown:
        parser.error(f"Not in the suite: {sorted(unknown)}")
    bases = {path: load_artifact(path, "imitator")[1] for path in args.imitator}
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    failed = False
    for case in cases:
        if args.case and case.case_id not in args.case:
            continue
        root = output / case.case_id.replace("/", "-").replace("+", "_")
        root.mkdir(exist_ok=True)
        result = dict(case_id=case.case_id, task_id=case.task_id, task=str(case.task), status="failed")
        prepared, deployment = root / "prepared", root / "deployment"
        if not (prepared / "prepared.json").is_file() and run("prepare", [
                "--task", str(case.task), "--output", str(prepared), "--iterations", str(args.iterations),
                "--device", args.device], root / "prepare.log"):
            result["reason"] = "prepare failed, see prepare.log"
        elif not (deployment / "deployment.json").is_file():
            signatures = load_artifact(prepared, "prepared")[1]["compatibility"]
            chosen = {side: next((path for path, base in bases.items()
                                  if base["side"] == side and same_signature(base["compatibility"], signature)), None)
                      for side, signature in signatures.items()}
            missing = [side for side, path in chosen.items() if path is None]
            if missing:
                result["reason"] = f"no given base matches the {' and '.join(missing)} hand physics of this task"
            elif run("train", ["--prepared", str(prepared), "--output", str(deployment), "--device", args.device,
                               *(item for side, path in chosen.items() for item in (f"--imitator-{side}", path)),
                               *train_options], root / "train.log"):
                result["reason"] = "training failed, see train.log"
        if "reason" in result:
            failed = True
        else:
            result["status"] = "completed"
        print(f"{case.case_id}: {result.get('reason', 'completed')}", flush=True)
        write_json(root / "result.json", result)
    sys.exit(failed)


if __name__ == "__main__":
    main()
