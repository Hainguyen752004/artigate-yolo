"""Run Paper 1 training rows one after another, each with its own name <row>_s<seed>.

    python scripts/run_plan.py --rows A4 --seeds 0                 # milestone 1
    python scripts/run_plan.py --rows A1 A1s --seeds 0             # milestone 2 (with A4_s0)
    python scripts/run_plan.py --rows A0 A2 A3 A5 A7 Y26 A1v7 A1v17 --seeds 0     # milestone 3, then:
    python scripts/run_plan.py --rows all --seeds 1 2

Each run is scripts/train.py with the row's model, data and p_synth from configs/runs.yaml. Before
anything starts, HEAD must be the commit of --tag and the code must have no uncommitted change, so
every provenance.json points at the same code. A run folder that already has provenance.json is
finished and skipped; a folder without it is an interrupted run and stops the plan (resume it with
Ultralytics or move it away first). Seeds are the outer loop: all rows of seed 1 before seed 2.
"""
import argparse
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from artigate.provenance import git_state

RUNS_YAML = ROOT / "configs" / "runs.yaml"


def load_rows(path=RUNS_YAML):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["rows"]


def train_arguments(row, spec, seed):
    """scripts/train.py arguments for one run of a row."""
    arguments = ["--model", spec["model"], "--data", spec["data"], "--seed", str(seed), "--name", f"{row}_s{seed}"]
    if "p_synth" in spec:
        arguments += ["--p-synth", str(spec["p_synth"])]
    return arguments


def plan(rows, seeds, specs):
    unknown = [row for row in rows if row not in specs]
    if unknown:
        raise SystemExit(f"unknown rows {unknown}; configs/runs.yaml has {list(specs)}")
    return [(f"{row}_s{seed}", train_arguments(row, specs[row], seed)) for seed in seeds for row in rows]


def tag_commit(tag, root=ROOT):
    result = subprocess.run(["git", "rev-parse", f"{tag}^{{commit}}"], cwd=root, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"tag {tag!r} does not exist; tag the training commit first (git tag {tag} && git push origin {tag})")
    return result.stdout.strip()


def check_code_version(tag, root=ROOT):
    """Refuse to train unless HEAD is the tagged commit and the code is unchanged."""
    expected, state = tag_commit(tag, root), git_state(root)
    if state is None:
        raise SystemExit("not a git checkout; official runs must be traceable to a commit")
    if state["commit"] != expected:
        raise SystemExit(f"HEAD is {state['commit'][:8]}, but {tag} is {expected[:8]}: git checkout {tag}")
    if state["dirty"]:
        raise SystemExit("the code has uncommitted changes (git status); official runs need a clean checkout")
    return expected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", nargs="+", required=True, help="row names from configs/runs.yaml, or 'all'")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--tag", default="train-v1", help="git tag every official run must be made from")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "segment")
    parser.add_argument("--dry-run", action="store_true", help="print the runs without checking the tag or training")
    args = parser.parse_args(argv)

    specs = load_rows()
    rows = list(specs) if args.rows == ["all"] else args.rows
    runs = plan(rows, args.seeds, specs)
    if not args.dry_run:
        commit = check_code_version(args.tag)
        print(f"[PLAN] code at {args.tag} = {commit[:8]}, clean")
    for name, arguments in runs:
        run_dir = args.project / name
        if (run_dir / "provenance.json").is_file():
            print(f"[PLAN] {name}: finished, skipped")
            continue
        if run_dir.exists():
            raise SystemExit(f"{run_dir} exists without provenance.json: an interrupted run. Resume or move it first.")
        command = [sys.executable, str(ROOT / "scripts" / "train.py"), *arguments, "--project", str(args.project)]
        print(f"[PLAN] {name}: {' '.join(arguments)}")
        if not args.dry_run:
            subprocess.run(command, check=True, cwd=ROOT)
    return runs


if __name__ == "__main__":
    main()
