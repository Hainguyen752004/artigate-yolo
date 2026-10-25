"""Turn image-level artifact annotations into the flags CSV read by the ArtiGate dataset.

Output columns: image_id,hair,ruler,marker,dark_corner (1 present, 0 absent, empty unknown).
Works with the Bissoto et al. annotations (github.com/alceubissoto/debiasing-skin) and with any CSV
whose column names contain hair / ruler / ink|mark / dark|vignet|corner. The column matching is
printed; check it once.

    python scripts/make_artifact_flags.py --src data/raw/bissoto/<file>.csv --out data/splits/isic2018_train_flags.csv
Optional: --holdout-frac 0.15 --holdout-out data/splits/holdout.txt draws an artifact-stratified list
of ids (only if the test set cannot be labelled with scripts/label_artifacts.py).
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import random
import re

RULES = {"hair": r"hair", "ruler": r"ruler", "marker": r"ink|mark", "dark_corner": r"dark|vignet|corner"}


def find_column(columns, pattern, used):
    return next((c for c in columns if re.search(pattern, c, re.I) and c not in used), None)


def to_flag(value):
    text = str(value).strip().lower()
    if text in ("1", "1.0", "true", "yes", "y"):
        return "1"
    if text in ("0", "0.0", "false", "no", "n"):
        return "0"
    try:
        return "1" if float(text) > 0 else "0"
    except ValueError:
        return ""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--id-col", help="column with the image name (auto-detected if omitted)")
    parser.add_argument("--holdout-frac", type=float, default=0.0)
    parser.add_argument("--holdout-out")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    with open(args.src, newline="", encoding="utf-8-sig") as stream:
        # The Bissoto annotation files are semicolon-separated.
        try:
            dialect = csv.Sniffer().sniff(stream.read(4096), delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        stream.seek(0)
        reader = csv.DictReader(stream, dialect=dialect)
        columns, rows = reader.fieldnames, list(reader)
    id_col = args.id_col or next((c for c in columns if re.fullmatch(r"(image|image_id|img|name|file|id)", c, re.I)), columns[0])
    print(f"id column: {id_col}")
    used, mapping = {id_col}, {}
    for name, pattern in RULES.items():
        column = find_column(columns, pattern, used)
        mapping[name] = column
        if column:
            used.add(column)
    out_rows = []
    for row in rows:
        record = {"image_id": Path(row[id_col]).stem}
        for name, column in mapping.items():
            record[name] = to_flag(row[column]) if column else ""
        out_rows.append(record)
    for name, column in mapping.items():
        present = sum(r[name] == "1" for r in out_rows)
        print(f"  {name:<12} -> {column or '(none: left unknown)':<22} present in {present}/{len(out_rows)}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_id", *RULES])
        writer.writeheader()
        writer.writerows(out_rows)
    print("wrote", args.out)

    if args.holdout_frac > 0:
        if not args.holdout_out:
            parser.error("--holdout-out is required with --holdout-frac")
        strata = defaultdict(list)
        for r in out_rows:
            strata["".join(r[n] or "?" for n in RULES)].append(r["image_id"])
        rng = random.Random(args.seed)
        held = []
        for key in sorted(strata):
            ids = sorted(strata[key])
            rng.shuffle(ids)
            held += ids[: round(len(ids) * args.holdout_frac)]
        Path(args.holdout_out).write_text("\n".join(sorted(held)) + "\n", encoding="utf-8")
        print(f"hold-out: {len(held)} ids -> {args.holdout_out}")


if __name__ == "__main__":
    main()
