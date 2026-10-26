"""Check that training runs were made from the tagged code with the locked schedule and the right row.

    python scripts/check_provenance.py --tag train-v1 runs/segment/A*_s*

For each run folder <row>_s<seed>: provenance.json exists; its commit is the commit of --tag and the
code was clean; the schedule equals configs/train/default.yaml (optimizer AdamW, lr0, epochs, batch,
no smoke-test fraction); Ultralytics' args.yaml agrees on optimizer, lr0, epochs and batch; and the
model, data, p_synth and seed are those of the row in configs/runs.yaml, and the preprocessing views
of the data export (metadata/export_summary.json) are the row's views. A4/A5 and A1/A1s/A1v7/A1v17
share a model YAML, so this is what tells them apart. Exits non-zero if any run fails.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from artigate.data.dataset_yaml import load_dataset_yaml

SCHEDULE_KEYS = ("optimizer", "lr0", "momentum", "warmup_bias_lr", "epochs", "batch", "imgsz")
DEFAULT_VIEWS = {"train": ["v1"], "eval": ["v1"]}


def _same_file(recorded, expected):
    return Path(str(recorded)).as_posix().endswith(Path(expected).as_posix())


def export_views(data_yaml):
    """{"train": [...], "eval": [...]} views of the export a dataset YAML points at, or None if unknown."""
    try:
        _, root = load_dataset_yaml(data_yaml)
        splits = json.loads((root / "metadata" / "export_summary.json").read_text(encoding="utf-8"))["splits"]
    except (OSError, KeyError, ValueError):
        return None
    evals = {tuple(splits[split]["views"]) for split in ("val", "test") if split in splits}
    return {"train": splits.get("train", {}).get("views"), "eval": list(evals.pop()) if len(evals) == 1 else sorted(evals)}


def check_run(run_dir, commit, schedule, rows, views_of=export_views):
    """Problems found in one run folder (an empty list means the run passes)."""
    run_dir = Path(run_dir)
    match = re.fullmatch(r"(.+)_s(\d+)", run_dir.name)
    if not match or match.group(1) not in rows:
        return [f"name {run_dir.name!r} is not <row>_s<seed> for a row of configs/runs.yaml"]
    row, seed = match.group(1), int(match.group(2))
    provenance_path = run_dir / "provenance.json"
    if not provenance_path.is_file():
        return ["no provenance.json (unfinished or not made by scripts/train.py)"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    problems = []
    git = provenance.get("git") or {}
    if git.get("commit") != commit:
        problems.append(f"commit {str(git.get('commit'))[:8]} is not the tag commit {commit[:8]}")
    if git.get("dirty", True):
        problems.append("code had uncommitted changes")
    hyp = provenance.get("hyp") or {}
    for key in SCHEDULE_KEYS:
        if hyp.get(key) != schedule.get(key):
            problems.append(f"hyp {key}={hyp.get(key)!r}, schedule has {schedule.get(key)!r}")
    if hyp.get("fraction", 1.0) != 1.0:
        problems.append(f"trained on fraction {hyp.get('fraction')} of the data")
    args_path = run_dir / "args.yaml"
    if args_path.is_file():
        used = yaml.safe_load(args_path.read_text(encoding="utf-8")) or {}
        for key in ("optimizer", "lr0", "epochs", "batch"):
            if used.get(key) != schedule.get(key):
                problems.append(f"args.yaml {key}={used.get(key)!r}, schedule has {schedule.get(key)!r}")
    else:
        problems.append("no args.yaml")
    spec = rows[row]
    if not _same_file(provenance.get("model_yaml", ""), spec["model"]):
        problems.append(f"model {provenance.get('model_yaml')} is not {spec['model']}")
    if not _same_file(provenance.get("data_yaml", ""), spec["data"]):
        problems.append(f"data {provenance.get('data_yaml')} is not {spec['data']}")
    overrides = ((provenance.get("artigate") or {}).get("data_overrides")) or {}
    if overrides.get("p_synth") != spec.get("p_synth"):
        problems.append(f"p_synth override {overrides.get('p_synth')!r}, row {row} needs {spec.get('p_synth')!r}")
    if provenance.get("seed") != seed:
        problems.append(f"seed {provenance.get('seed')} does not match the name (s{seed})")
    data_yaml = Path(str(provenance.get("data_yaml", "")))
    views = views_of(data_yaml if data_yaml.is_file() else ROOT / spec["data"])
    expected = spec.get("views", DEFAULT_VIEWS)
    if views is None:
        problems.append(f"cannot read the views of the data export behind {spec['data']}")
    elif views != expected:
        problems.append(f"data export views {views}, row {row} needs {expected}")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--tag", default="train-v1")
    parser.add_argument("--commit", default=None, help="expected commit (default: the commit of --tag)")
    args = parser.parse_args(argv)

    commit = args.commit or subprocess.run(["git", "rev-parse", f"{args.tag}^{{commit}}"], cwd=ROOT,
                                           capture_output=True, text=True, check=True).stdout.strip()
    schedule = yaml.safe_load((ROOT / "configs" / "train" / "default.yaml").read_text(encoding="utf-8"))
    rows = yaml.safe_load((ROOT / "configs" / "runs.yaml").read_text(encoding="utf-8"))["rows"]
    failed = 0
    for run_dir in sorted(path for path in args.runs if path.is_dir()):
        problems = check_run(run_dir, commit, schedule, rows)
        failed += bool(problems)
        print(f"{'OK  ' if not problems else 'FAIL'} {run_dir.name}" + "".join(f"\n     - {p}" for p in problems))
    print(f"{failed} of {len([p for p in args.runs if p.is_dir()])} runs failed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
