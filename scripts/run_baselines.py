"""Train the non-YOLO baselines one after another (CLAUDE.md order):
EGE-UNet s0-2, MALUNet s0-2, U-Net s0-2 (input 256), then the sensitivity runs malunet640_s0 and
egeunet640_s0 (input 640, configs/baselines/<model>_640.yaml).

    python scripts/run_baselines.py --project ~/iDragonCloud/paperArtigate/runs/baselines
    python scripts/run_baselines.py --models egeunet --seeds 0 --no-sensitivity --dry-run

Each run is scripts/train_seg_baseline.py --model <m> --seed <s> (plus --config for the 640 runs). Before
anything starts, HEAD must be the commit of --tag (default baselines-v1) and the code must have no
uncommitted change, as for the YOLO runs.
A run folder with provenance.json is finished and skipped; one with last.pt but no provenance.json is
continued with --resume. nnU-Net 2D runs through its own commands (scripts/prepare_nnunet.py).
"""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_plan import check_code_version

ORDER = ("egeunet", "malunet", "unet")
SENSITIVITY = ("malunet", "egeunet")  # 640-input runs, seed 0, after the main runs


def plan(models, seeds, sensitivity=True):
    """(name, model, seed, recipe or None): main runs with models outer and seeds inner, then the 640 runs."""
    unknown = [m for m in models if m not in ORDER]
    if unknown:
        raise SystemExit(f"unknown baselines {unknown}; expected {list(ORDER)}")
    runs = [(f"{m}_s{s}", m, s, None) for m in ORDER if m in models for s in seeds]
    if sensitivity:
        runs += [(f"{m}640_s0", m, 0, ROOT / "configs" / "baselines" / f"{m}_640.yaml") for m in SENSITIVITY if m in models]
    return runs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(ORDER))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--tag", default="baselines-v1")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "baselines")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-sensitivity", action="store_true", help="skip malunet640_s0 and egeunet640_s0")
    parser.add_argument("--dry-run", action="store_true", help="print the runs without checking the tag or training")
    args = parser.parse_args(argv)

    runs = plan(args.models, args.seeds, not args.no_sensitivity)
    if not args.dry_run:
        commit = check_code_version(args.tag)
        print(f"[PLAN] code at {args.tag} = {commit[:8]}, clean", flush=True)
    for name, model, seed, recipe in runs:
        run_dir = args.project / name
        if (run_dir / "provenance.json").is_file():
            print(f"[PLAN] {name}: finished, skipped", flush=True)
            continue
        command = [sys.executable, str(ROOT / "scripts" / "train_seg_baseline.py"), "--model", model, "--seed", str(seed),
                   "--project", str(args.project), "--name", name]
        if recipe is not None:
            command += ["--config", str(recipe)]
        if (run_dir / "last.pt").is_file():
            command.append("--resume")
        if args.device is not None:
            command += ["--device", args.device]
        print(f"[PLAN] {name}: {' '.join(command[2:])}", flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True, cwd=ROOT)
    return runs


if __name__ == "__main__":
    main()
