"""Agreement between two artifact annotators, and the final test-set artifact labels.

1. Agreement on the common subset (Cohen's kappa per type, positive and negative specific agreement, each
   with a 95% bootstrap interval over images; raw agreement, prevalence), plus a list of the disagreements
   to resolve by discussion:
       python scripts/artifact_agreement.py --a1 data/splits/isic2018_test_artifacts_a1.csv \
           --a2 data/splits/isic2018_test_artifacts_a2.csv --out runs/agreement/isic2018_test
   writes <out>.json, <out>.md and <out>_disagreements.csv (fill its "resolved" column with 0/1).

2. Final labels = annotator 1, with the resolved values on the disagreements:
       python scripts/artifact_agreement.py --a1 ... --a2 ... --out runs/agreement/isic2018_test \
           --resolved runs/agreement/isic2018_test_disagreements.csv --final data/splits/isic2018_test_artifacts.csv
   The final file (image_id, hair, ruler, marker, dark_corner) is what compare_family.py --strata reads.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

TYPES = ("hair", "ruler", "marker", "dark_corner")


def read_labels(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return {row["image_id"]: {t: int(row[t]) for t in TYPES} for row in csv.DictReader(stream)}


def cohen_kappa(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    if a.size == 0:
        return float("nan")
    p_o = float((a == b).mean())
    p_e = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return float("nan") if p_e >= 1.0 else (p_o - p_e) / (1.0 - p_e)


def specific_agreement(a, b):
    """(positive, negative) specific agreement: 2*both_1 / (2*both_1 + disagreements), and the same for 0.

    Unlike kappa they stay interpretable for a rare type, where kappa's interval is very wide.
    """
    a, b = np.asarray(a, int), np.asarray(b, int)
    both_pos, both_neg, differ = int(((a == 1) & (b == 1)).sum()), int(((a == 0) & (b == 0)).sum()), int((a != b).sum())
    pos = 2 * both_pos / (2 * both_pos + differ) if 2 * both_pos + differ else float("nan")
    neg = 2 * both_neg / (2 * both_neg + differ) if 2 * both_neg + differ else float("nan")
    return float(pos), float(neg)


def _interval(values):
    values = np.array([v for v in values if np.isfinite(v)])
    if not values.size:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def agreement(a1, a2, n_boot=2000, seed=0):
    common = sorted(set(a1) & set(a2))
    rng = np.random.default_rng(seed)
    report = {"n_common": len(common), "n_a1": len(a1), "types": {}}
    for t in TYPES:
        x = np.array([a1[i][t] for i in common], int)
        y = np.array([a2[i][t] for i in common], int)
        kappas, positives, negatives = [], [], []
        for _ in range(n_boot):
            idx = rng.integers(0, len(common), len(common))
            kappas.append(cohen_kappa(x[idx], y[idx]))
            pos, neg = specific_agreement(x[idx], y[idx])
            positives.append(pos)
            negatives.append(neg)
        (ci_low, ci_high), pos, neg = _interval(kappas), *specific_agreement(x, y)
        report["types"][t] = {
            "kappa": cohen_kappa(x, y),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "positive_agreement": pos,
            "positive_agreement_ci": _interval(positives),
            "negative_agreement": neg,
            "negative_agreement_ci": _interval(negatives),
            "positives_a1_common": int(x.sum()),
            "positives_a2_common": int(y.sum()),
            "raw_agreement": float((x == y).mean()) if len(common) else float("nan"),
            "prevalence_a1_all": float(np.mean([a1[i][t] for i in a1])),
            "positives_a1_all": int(sum(a1[i][t] for i in a1)),
            "disagreements": int((x != y).sum()),
        }
    any_art = [int(any(a1[i][t] for t in TYPES)) for i in a1]
    report["any_artifact_a1_all"] = int(sum(any_art))
    report["no_artifact_a1_all"] = len(any_art) - int(sum(any_art))
    return report, common


def disagreements(a1, a2, common):
    return [{"image_id": i, "type": t, "a1": a1[i][t], "a2": a2[i][t], "resolved": ""}
            for i in common for t in TYPES if a1[i][t] != a2[i][t]]


def final_labels(a1, resolved_rows):
    final = {i: dict(v) for i, v in a1.items()}
    missing = [r for r in resolved_rows if str(r.get("resolved", "")).strip() not in ("0", "1")]
    if missing:
        raise SystemExit(f"{len(missing)} disagreements have no resolved value (0/1), e.g. {missing[0]}")
    for r in resolved_rows:
        final[r["image_id"]][r["type"]] = int(r["resolved"])
    return final


def write_labels(path, labels):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_id", *TYPES])
        for i in sorted(labels):
            writer.writerow([i, *(labels[i][t] for t in TYPES)])


def markdown(report):
    lines = [f"Common subset: {report['n_common']} images; annotator 1 labelled {report['n_a1']} images "
             f"({report['any_artifact_a1_all']} with any artifact, {report['no_artifact_a1_all']} with none).", "",
             "| Type | Positives (a1, all) | Positives on subset (a1 / a2) | Raw agreement | Cohen's kappa [95% CI] "
             "| Positive agreement [95% CI] | Negative agreement [95% CI] | Disagreements |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for t, r in report["types"].items():
        pa, na = r["positive_agreement_ci"], r["negative_agreement_ci"]
        lines.append(f"| {t} | {r['positives_a1_all']} ({100 * r['prevalence_a1_all']:.1f}%) | "
                     f"{r['positives_a1_common']} / {r['positives_a2_common']} | {100 * r['raw_agreement']:.1f}% | "
                     f"{r['kappa']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] | "
                     f"{r['positive_agreement']:.3f} [{pa[0]:.3f}, {pa[1]:.3f}] | "
                     f"{r['negative_agreement']:.3f} [{na[0]:.3f}, {na[1]:.3f}] | {r['disagreements']} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a1", type=Path, required=True)
    ap.add_argument("--a2", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output prefix")
    ap.add_argument("--resolved", type=Path, help="disagreements CSV with the 'resolved' column filled")
    ap.add_argument("--final", type=Path, help="write the final labels here (needs --resolved)")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args(argv)

    a1, a2 = read_labels(args.a1), read_labels(args.a2)
    report, common = agreement(a1, a2, args.n_boot)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    text = markdown(report)
    args.out.with_suffix(".md").write_text(text, encoding="utf-8")
    rows = disagreements(a1, a2, common)
    dis_path = args.out.parent / f"{args.out.name}_disagreements.csv"
    if not dis_path.exists() or args.resolved is None:
        with dis_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["image_id", "type", "a1", "a2", "resolved"])
            writer.writeheader()
            writer.writerows(rows)
    print(text)
    if args.final:
        if args.resolved is None:
            ap.error("--final needs --resolved")
        with args.resolved.open(newline="", encoding="utf-8") as stream:
            resolved = list(csv.DictReader(stream))
        write_labels(args.final, final_labels(a1, resolved))
        print(f"wrote final labels to {args.final}")
    return report


if __name__ == "__main__":
    main()
