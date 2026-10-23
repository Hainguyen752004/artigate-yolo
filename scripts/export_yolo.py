"""Export a split manifest to a YOLO-seg dataset under data/yolo/.

Training samples get --train-views (e.g. v1,v7 for the ICEEIS two-view baseline); val and test
get --eval-views. The manifest is checked for duplicate IDs, groups spanning splits and ISIC IDs
shared between splits before anything is written.
"""
import argparse
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from artigate.data.export import check_manifest, export_split, read_manifest, split_paths
from artigate.data.views import VIEWS
from artigate.provenance import sha256_file, write_json


def _views(value):
    views = tuple(view.strip() for view in value.split(",") if view.strip())
    unknown = set(views) - set(VIEWS)
    if not views or unknown:
        raise argparse.ArgumentTypeError(f"views must be a comma list drawn from {VIEWS}")
    return views


def _project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=_project_path, required=True)
    parser.add_argument("--out", type=_project_path, required=True, help="dataset root, e.g. data/yolo/isic2018")
    parser.add_argument("--raw-root", type=_project_path, default=ROOT / "data" / "raw")
    parser.add_argument("--train-views", type=_views, default=("v1",))
    parser.add_argument("--eval-views", type=_views, default=("v1",))
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--overwrite", action="store_true", help="replace splits that were already exported")
    args = parser.parse_args(argv)

    rows = read_manifest(args.manifest)
    check_manifest(rows)
    by_split = {}
    for row in rows:
        by_split.setdefault(row["split"], []).append(row)
    existing = [path for split in by_split for path in split_paths(args.out, split) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"already exported (pass --overwrite): {[str(path) for path in existing]}")
    for path in existing:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    summary = {"manifest": str(args.manifest), "manifest_sha256": sha256_file(args.manifest),
               "raw_root": str(args.raw_root), "size": args.size, "splits": {}}
    for split, split_rows in sorted(by_split.items()):
        views = args.train_views if split == "train" else args.eval_views
        result = export_split(split_rows, args.raw_root, args.out, split, views, args.size)
        summary["splits"][split] = {"views": list(views), **result}
        if result["empty_labels"]:
            print(f"[WARN] {split}: {len(result['empty_labels'])} images have no lesion polygon")
    write_json(args.out / "metadata" / "export_summary.json", summary)
    print(f"[INFO] Exported {args.manifest.name} to {args.out}")
    return summary


if __name__ == "__main__":
    main()
