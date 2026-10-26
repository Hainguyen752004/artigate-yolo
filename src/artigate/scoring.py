"""Test-set scoring rules shared by scripts/eval.py and scripts/eval_seg_baseline.py.

A test split (ISIC 2018 or an external set) is scored once per checkpoint, dataset and mask rule:
every test scoring is appended to <project>/test_scoring_log.jsonl, and a second scoring of the same
checkpoint (by its sha256) is refused, whatever the output name says. A rescore needs a reason, goes
to <name>_rescore<k> and is noted in rescore_log.txt of the original folder. Validation and
training splits can be rerun freely.

No test split is scored before the ISIC 2018 test-set artifact labels
(data/splits/isic2018_test_artifacts.csv, docs/annotation_test_artifacts.md) are committed: the
paper states that they were made before any model saw the test set.
"""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess

from artigate.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[2]
ISIC2018_DATA = ROOT / "configs" / "data" / "isic2018.yaml"
TEST_ARTIFACT_LABELS = ROOT / "data" / "splits" / "isic2018_test_artifacts.csv"
ARTIFACT_TYPES = ("hair", "ruler", "marker", "dark_corner")


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)


def committed_test_labels(path=TEST_ARTIFACT_LABELS, root=ROOT):
    """Check that the test-set artifact labels exist, are valid and are committed unchanged.

    Returns {path, commit, commit_time, sha256, n_images, image_ids}; refuses (SystemExit) otherwise,
    because no test split may be scored before these labels are fixed in git.
    """
    path, root = Path(path), Path(root)
    how = "see docs/annotation_test_artifacts.md"
    if not path.is_file():
        raise SystemExit(f"test scoring is locked: {path} does not exist yet. Annotate the test set and commit "
                         f"the labels first ({how}).")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or [c for c in ("image_id", *ARTIFACT_TYPES) if c not in reader.fieldnames]:
            raise SystemExit(f"{path} needs the columns image_id,{','.join(ARTIFACT_TYPES)} ({how})")
        rows = list(reader)
    bad = [r["image_id"] for r in rows if any(str(r[t]).strip() not in ("0", "1") for t in ARTIFACT_TYPES)]
    if not rows or bad:
        raise SystemExit(f"{path} must hold one 0/1 row per image; invalid rows: {bad[:5] or 'none (empty file)'}")
    try:
        rel = str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        raise SystemExit(f"{path} is outside the repository {root}; test scoring needs the committed copy") from None
    if _git(root, "ls-files", "--error-unmatch", rel).returncode != 0:
        raise SystemExit(f"test scoring is locked: {rel} is not committed. Commit it first ({how}).")
    if _git(root, "diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        raise SystemExit(f"test scoring is locked: {rel} differs from the committed version. Commit or restore it.")
    log = _git(root, "log", "-1", "--format=%H %cI", "--", rel).stdout.split()
    return {"path": rel, "commit": log[0] if log else None, "commit_time": log[1] if len(log) > 1 else None,
            "sha256": sha256_file(path), "n_images": len(rows), "image_ids": {r["image_id"] for r in rows}}


def test_label_record(data_yaml, image_stems):
    """committed_test_labels() for a test scoring, also checking that every ISIC 2018 test image is labelled.

    Returns the record without the id set, for eval_summary.json and the scoring log.
    """
    labels = committed_test_labels()
    if Path(data_yaml).resolve() == ISIC2018_DATA.resolve():
        unlabelled = [s for s in image_stems if re.sub(r"_v\d+$", "", s) not in labels["image_ids"]]
        if unlabelled:
            raise SystemExit(f"{len(unlabelled)} ISIC 2018 test images have no artifact labels, e.g. {unlabelled[:3]}")
    return {k: v for k, v in labels.items() if k != "image_ids"}


def default_name(weights, data_yaml, split, mask_rule="top1", run=None):
    """<run>_<data>_<split>[_union]; run = the run folder of the checkpoint unless given."""
    if run is None:
        weights = Path(weights)
        if weights.parent.name == "weights":  # Ultralytics: runs/segment/<run>/weights/best.pt
            run = weights.parent.parent.name
        elif weights.stem in ("best", "last"):  # baselines: runs/baselines/<run>/best.pt
            run = weights.parent.name
        else:
            run = weights.stem
    data_yaml = Path(data_yaml)
    # generated sets keep one data.yaml per folder (e.g. isic2018_sev/hair_s3/data.yaml): name them by folder
    data = data_yaml.parent.name if data_yaml.stem in ("data", "dataset") else data_yaml.stem
    return f"{run}_{data}_{split}" + ("_union" if mask_rule == "union" else "")


def _scoring_log(project):
    return Path(project) / "test_scoring_log.jsonl"


def test_scorings(project, weights_sha, data_yaml, mask_rule):
    """Earlier test scorings of this checkpoint on this dataset with this mask rule."""
    log = _scoring_log(project)
    if not log.is_file():
        return []
    entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = str(Path(data_yaml).resolve())
    return [e for e in entries if e["weights_sha256"] == weights_sha and e["data_yaml"] == data and e["mask_rule"] == mask_rule]


def output_dir(project, name, split, weights_sha, data_yaml, mask_rule, rescore_reason=None):
    """Folder this evaluation writes to, and whether an existing folder may be reused.

    Validation and training splits reuse <project>/<name>. A test split gets <project>/<name> only if
    neither that folder nor an earlier scoring of the checkpoint exists; otherwise a non-empty
    rescore_reason is required and the results go to a fresh <name>_rescore<k>.
    """
    project = Path(project)
    save_dir = project / name
    if split != "test":
        return save_dir, True
    earlier = test_scorings(project, weights_sha, data_yaml, mask_rule)
    if not earlier and not save_dir.exists():
        return save_dir, False
    if not (rescore_reason or "").strip():
        where = earlier[0]["save_dir"] if earlier else str(save_dir)
        raise SystemExit(f"this checkpoint was already scored on the test split ({where}); test sets are scored "
                         f"once. Pass --rescore-reason \"...\" to score again into a new folder.")
    k = 1
    while (project / f"{name}_rescore{k}").exists():
        k += 1
    return project / f"{name}_rescore{k}", False


def record_test_scoring(project, name, save_dir, weights, weights_sha, data_yaml, mask_rule, rescore_reason=None,
                        labels=None):
    """Append to the scoring log; for a rescore, also note the reason in the original folder."""
    timestamp = datetime.now(timezone.utc).isoformat()
    entry = {"timestamp_utc": timestamp, "weights": str(weights), "weights_sha256": weights_sha,
             "data_yaml": str(Path(data_yaml).resolve()), "split": "test", "mask_rule": mask_rule,
             "save_dir": str(save_dir), "rescore_reason": rescore_reason,
             "test_artifact_labels_commit": (labels or {}).get("commit")}
    _scoring_log(project).parent.mkdir(parents=True, exist_ok=True)
    with _scoring_log(project).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry) + "\n")
    if rescore_reason:
        original = Path(project) / name
        original.mkdir(parents=True, exist_ok=True)
        with (original / "rescore_log.txt").open("a", encoding="utf-8") as stream:
            stream.write(f"{timestamp}\t{Path(save_dir).name}\t{rescore_reason}\n")
