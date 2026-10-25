"""Paired comparisons of several models against one reference, averaged over seeds, Holm-adjusted.

This is the analysis of the paper's tables: for every model, the per-image metric is first averaged
over its seeds (one per_image_metrics.csv from scripts/eval.py per seed); each model is then
compared with the reference on the same images (mean paired difference, 95% percentile bootstrap
interval, two-sided Wilcoxon signed-rank test with zero differences discarded). Holm's procedure
adjusts the p-values of all rows produced by one call, per metric, so one call = one family.

    # ablation family: every row against A4 on the full test set
    python scripts/compare_family.py --reference A4=runs/eval/A4_s0/per_image_metrics.csv,runs/eval/A4_s1/...,... \
        --model A1=... --model A1s=... --model A2=... --out runs/stats/ablation
    # primary family: A4 against A1 on the artifact strata
    python scripts/compare_family.py --reference A1=... --model A4=... \
        --strata data/splits/isic2018_test_artifacts.csv \
        --subsets any_artifact,with_hair,with_ruler,with_marker,with_dark_corner,no_artifact --out runs/stats/primary

Sensitivity analyses: --exclude data/splits/isic2018_test_near_duplicates.csv --exclude-verdicts
same_photo,same_lesion leaves out the test images that duplicate a training image.

Writes <out>.json and <out>.md. Differences are model minus reference; W/T/L counts the images on
which the model is better / equal / worse than the reference (lower is better for HD95).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from artigate.metrics import METRIC_NAMES  # noqa: E402
from artigate.provenance import write_json  # noqa: E402
from artigate.stats import bootstrap_ci, holm_adjust, paired_comparison  # noqa: E402
from bootstrap_ci import _sample_id, read_rows, strata_subsets  # noqa: E402

LOWER_IS_BETTER = {"hd95"}


def read_exclusions(path, verdicts):
    """Sample ids to leave out (e.g. test images that duplicate a training image).

    ``path`` is a CSV with an ``image_id`` column; if it has a ``verdict`` column, only rows whose
    verdict is in ``verdicts`` are used (all rows when ``verdicts`` is empty).
    """
    import csv

    keep = {v for v in verdicts if v}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {_sample_id(r["image_id"]) for r in rows if not keep or "verdict" not in r or r["verdict"] in keep}


def parse_spec(spec):
    name, sep, paths = spec.partition("=")
    if not sep or not name or not paths:
        raise argparse.ArgumentTypeError(f"expected NAME=csv[,csv...], got {spec!r}")
    return name, [Path(p) for p in paths.split(",") if p]


def seed_average(paths, metrics):
    """image_id -> {metric: mean over seeds}, plus the per-seed means of each metric."""
    runs = [read_rows(p) for p in paths]
    ids = set(runs[0])
    for path, rows in zip(paths[1:], runs[1:]):
        if set(rows) != ids:
            raise ValueError(f"{path} covers different images than {paths[0]}")
    per_image = {i: {m: float(np.mean([float(r[i][m]) for r in runs])) for m in metrics} for i in ids}
    per_seed = {m: [float(np.mean([float(r[i][m]) for i in ids])) for r in runs] for m in metrics}
    return per_image, per_seed


def compare(reference, models, metrics, subsets, n_boot=2000, seed=0, min_stratum=10):
    ref_name, ref_paths = reference
    ref_images, ref_seeds = seed_average(ref_paths, metrics)
    rows = []
    summaries = {ref_name: {"n_seeds": len(ref_paths), "per_seed_means": ref_seeds}}
    for name, paths in models:
        images, per_seed = seed_average(paths, metrics)
        if set(images) != set(ref_images):
            raise ValueError(f"{name} and {ref_name} cover different images")
        summaries[name] = {"n_seeds": len(paths), "per_seed_means": per_seed}
        for subset, ids in subsets.items():
            ids = sorted(ids)
            if len(ids) < max(1, min_stratum if subset != "all" else 1):
                continue  # too few images to test; reported in "skipped"
            for metric in metrics:
                model_values = [images[i][metric] for i in ids]
                ref_values = [ref_images[i][metric] for i in ids]
                paired = paired_comparison(model_values, ref_values, n_boot=n_boot, seed=seed)
                diff = np.asarray(model_values) - np.asarray(ref_values)
                better, worse = (diff < 0, diff > 0) if metric in LOWER_IS_BETTER else (diff > 0, diff < 0)
                rows.append({
                    "model": name, "reference": ref_name, "subset": subset, "metric": metric, "n": len(ids),
                    "model_mean": bootstrap_ci(model_values, n_boot, seed)["point"],
                    "reference_mean": bootstrap_ci(ref_values, n_boot, seed)["point"], **paired,
                    "wins": int(better.sum()), "ties": int((diff == 0).sum()), "losses": int(worse.sum()),
                })
    for metric in metrics:
        family = [r for r in rows if r["metric"] == metric]
        for row, p_holm in zip(family, holm_adjust([r["wilcoxon_p"] for r in family])):
            row["p_holm"] = p_holm
            row["family_size"] = len(family)
    return {"reference": ref_name, "n_boot": n_boot, "seed": seed, "models": summaries, "rows": rows}


def markdown(report):
    lines = [f"Reference: {report['reference']} (difference = model - reference; Holm within each metric)", "",
             "| Model | Subset | Metric | n | Model | Ref. | Diff [95% CI] | p | p Holm | W/T/L |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in report["rows"]:
        lines.append(f"| {r['model']} | {r['subset']} | {r['metric']} | {r['n']} | {r['model_mean']:.4f} | "
                     f"{r['reference_mean']:.4f} | {r['mean_difference']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | "
                     f"{r['wilcoxon_p']:.3g} | {r['p_holm']:.3g} | {r['wins']}/{r['ties']}/{r['losses']} |")
    lines += ["", "Per-seed means (mean +- SD over seeds):", ""]
    for name, s in report["models"].items():
        parts = [f"{m} {np.mean(v):.4f} +- {np.std(v, ddof=1) if len(v) > 1 else 0.0:.4f}" for m, v in s["per_seed_means"].items()]
        lines.append(f"- {name} ({s['n_seeds']} seeds): " + ", ".join(parts))
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=parse_spec, required=True, help="NAME=csv[,csv...] (one csv per seed)")
    ap.add_argument("--model", type=parse_spec, action="append", required=True, help="NAME=csv[,csv...], repeatable")
    ap.add_argument("--metrics", default=",".join(METRIC_NAMES))
    ap.add_argument("--strata", type=Path, default=None, help="image-level artifact flags CSV")
    ap.add_argument("--subsets", default="all", help="comma list of subsets; 'all' = every test image")
    ap.add_argument("--min-stratum", type=int, default=10, help="skip subsets with fewer images")
    ap.add_argument("--exclude", type=Path, default=None,
                    help="CSV of image ids to leave out of every subset (sensitivity analyses)")
    ap.add_argument("--exclude-verdicts", default="",
                    help="with --exclude: comma list of verdicts to use, e.g. same_photo,same_lesion (default: all rows)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True, help="output path without extension")
    args = ap.parse_args(argv)

    metrics = [m for m in args.metrics.split(",") if m]
    ref_ids = sorted(read_rows(args.reference[1][0]))
    wanted = [s for s in args.subsets.split(",") if s]
    available = {"all": ref_ids}
    if args.strata is not None:
        available.update(strata_subsets(args.strata, ref_ids)[0])
    missing = [s for s in wanted if s not in available]
    if missing:
        ap.error(f"unknown subsets {missing}; available: {sorted(available)} (strata need --strata)")
    chosen = {s: available[s] for s in wanted}
    excluded = set()
    if args.exclude is not None:
        excluded = read_exclusions(args.exclude, args.exclude_verdicts.split(","))
        chosen = {s: [i for i in ids if _sample_id(i) not in excluded] for s, ids in chosen.items()}
    report = compare(args.reference, args.model, metrics, chosen, args.n_boot, args.seed, args.min_stratum)
    report["skipped"] = {s: len(ids) for s, ids in chosen.items() if s != "all" and len(ids) < args.min_stratum}
    report["strata_csv"] = str(args.strata) if args.strata else None
    report["excluded"] = {"csv": str(args.exclude) if args.exclude else None, "verdicts": args.exclude_verdicts,
                          "n_listed": len(excluded),
                          "n_removed": len([i for i in ref_ids if _sample_id(i) in excluded])}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out.with_suffix(".json"), report)
    text = markdown(report)
    args.out.with_suffix(".md").write_text(text, encoding="utf-8")
    print(text)
    return report


if __name__ == "__main__":
    main()
