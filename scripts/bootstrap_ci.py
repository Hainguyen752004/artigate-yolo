"""Bootstrap CIs for one evaluation, optionally paired against a baseline evaluation.

Reads per_image_metrics.csv written by scripts/eval.py. With --baseline, both files must cover
the same images; the report adds the mean paired difference, its bootstrap CI and a Wilcoxon
signed-rank p-value per metric. Ultralytics mAP is a point estimate and is not bootstrapped here.
"""
import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from artigate.metrics import METRIC_NAMES
from artigate.provenance import write_json
from artigate.stats import bootstrap_ci, paired_comparison


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = {row["image_id"]: row for row in csv.DictReader(stream)}
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def column(rows, image_ids, metric):
    return [float(rows[image_id][metric]) for image_id in image_ids]


def run(metrics_csv, baseline_csv=None, n_boot=2000, seed=0):
    rows = read_rows(metrics_csv)
    image_ids = sorted(rows)
    report = {
        "metrics_csv": str(metrics_csv),
        "n_images": len(image_ids),
        "n_boot": n_boot,
        "seed": seed,
        "ci_method": "image-level nonparametric percentile bootstrap, 95%",
        "metrics": {metric: bootstrap_ci(column(rows, image_ids, metric), n_boot, seed) for metric in METRIC_NAMES},
    }
    if baseline_csv is not None:
        baseline = read_rows(baseline_csv)
        if set(baseline) != set(rows):
            raise ValueError("baseline and model evaluations cover different images")
        report["baseline_csv"] = str(baseline_csv)
        report["paired"] = {
            metric: paired_comparison(column(rows, image_ids, metric), column(baseline, image_ids, metric), n_boot, seed)
            for metric in METRIC_NAMES
        }
    return report


def markdown(report):
    lines = ["| Metric | Mean | 95% bootstrap CI |", "|---|---:|---:|"]
    for metric, values in report["metrics"].items():
        lines.append(f"| {metric} | {values['point']:.4f} | [{values['ci_low']:.4f}, {values['ci_high']:.4f}] |")
    if "paired" in report:
        lines += ["", f"Paired against `{report['baseline_csv']}` (model - baseline):", "",
                  "| Metric | Mean diff | 95% bootstrap CI | Wilcoxon p |", "|---|---:|---:|---:|"]
        for metric, values in report["paired"].items():
            lines.append(
                f"| {metric} | {values['mean_difference']:+.4f} | [{values['ci_low']:+.4f}, {values['ci_high']:+.4f}] "
                f"| {values['wilcoxon_p']:.3g} |"
            )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("metrics_csv", type=Path)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="output stem (default: next to metrics_csv)")
    args = parser.parse_args(argv)
    report = run(args.metrics_csv, args.baseline, args.n_boot, args.seed)
    if args.out is not None:
        stem = args.out
    elif args.baseline is not None:
        stem = args.metrics_csv.parent / f"bootstrap_vs_{args.baseline.parent.name}"
    else:
        stem = args.metrics_csv.parent / "bootstrap_ci"
    write_json(stem.with_suffix(".json"), report)
    stem.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    return report


if __name__ == "__main__":
    main()
