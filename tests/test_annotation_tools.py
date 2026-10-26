"""Test-set artifact annotation: image selection for the labelling tool and annotator agreement."""
import csv

import pytest


def _write(path, rows, header):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def test_image_list_from_manifest_and_subset(tmp_path, load_script):
    tool = load_script("label_artifacts")
    _write(tmp_path / "m.csv", [["ISIC_2", "test", "a/ISIC_2.jpg", "", ""], ["ISIC_1", "test", "a/ISIC_1.jpg", "", ""],
                                ["ISIC_0", "train", "a/ISIC_0.jpg", "", ""]], ["sample_id", "split", "image", "mask", "group"])
    _write(tmp_path / "ids.csv", [["ISIC_2_v1"]], ["image_id"])
    items = tool.image_list(manifest=tmp_path / "m.csv", split="test", raw_root=tmp_path / "raw")
    assert [i for i, _ in items] == ["ISIC_1", "ISIC_2"] and items[0][1] == tmp_path / "raw" / "a" / "ISIC_1.jpg"
    subset = tool.image_list(manifest=tmp_path / "m.csv", split="test", raw_root=tmp_path, ids=tmp_path / "ids.csv")
    assert [i for i, _ in subset] == ["ISIC_2"]


def test_agreement_kappa_disagreements_and_final_labels(tmp_path, load_script):
    script = load_script("artifact_agreement")
    header = ["image_id", "hair", "ruler", "marker", "dark_corner"]
    a1 = [[f"I{i}", int(i % 2 == 0), int(i % 3 == 0), 0, int(i < 5)] for i in range(20)]
    a2 = [[f"I{i}", int(i % 2 == 0), int(i % 3 == 0) if i != 3 else 0, 0, int(i < 5)] for i in range(10)]
    _write(tmp_path / "a1.csv", a1, header)
    _write(tmp_path / "a2.csv", a2, header)
    out = tmp_path / "agr" / "test"
    report = script.main(["--a1", str(tmp_path / "a1.csv"), "--a2", str(tmp_path / "a2.csv"), "--out", str(out), "--n-boot", "200"])
    assert report["n_common"] == 10 and report["n_a1"] == 20
    assert report["types"]["hair"]["kappa"] == pytest.approx(1.0)
    assert report["types"]["ruler"]["disagreements"] == 1 and report["types"]["ruler"]["kappa"] < 1.0
    dis = out.parent / "test_disagreements.csv"
    rows = list(csv.DictReader(dis.open(encoding="utf-8")))
    assert rows == [{"image_id": "I3", "type": "ruler", "a1": "1", "a2": "0", "resolved": ""}]
    with pytest.raises(SystemExit):
        script.main(["--a1", str(tmp_path / "a1.csv"), "--a2", str(tmp_path / "a2.csv"), "--out", str(out),
                     "--resolved", str(dis), "--final", str(tmp_path / "final.csv"), "--n-boot", "50"])
    _write(dis, [["I3", "ruler", 1, 0, 0]], ["image_id", "type", "a1", "a2", "resolved"])
    script.main(["--a1", str(tmp_path / "a1.csv"), "--a2", str(tmp_path / "a2.csv"), "--out", str(out),
                 "--resolved", str(dis), "--final", str(tmp_path / "final.csv"), "--n-boot", "50"])
    final = {r["image_id"]: r for r in csv.DictReader((tmp_path / "final.csv").open(encoding="utf-8"))}
    assert len(final) == 20 and final["I3"]["ruler"] == "0" and final["I6"]["ruler"] == "1"


def test_test_scoring_is_locked_until_the_labels_are_committed(tmp_path, load_script):
    import subprocess

    evaluate = load_script("eval")
    repo = tmp_path / "repo"
    labels = repo / "data" / "splits" / "isic2018_test_artifacts.csv"
    labels.parent.mkdir(parents=True)
    git = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)  # noqa: E731
    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    with pytest.raises(SystemExit, match="does not exist"):
        evaluate.committed_test_labels(labels, repo)
    header = ["image_id", "hair", "ruler", "marker", "dark_corner"]
    _write(labels, [["ISIC_1", 1, 0, 0, 0], ["ISIC_2", 0, 2, 0, 0]], header)
    with pytest.raises(SystemExit, match="invalid rows"):
        evaluate.committed_test_labels(labels, repo)
    _write(labels, [["ISIC_1", 1, 0, 0, 0], ["ISIC_2", 0, 1, 0, 0]], header)
    with pytest.raises(SystemExit, match="not committed"):
        evaluate.committed_test_labels(labels, repo)
    git("add", ".")
    git("commit", "-qm", "labels")
    info = evaluate.committed_test_labels(labels, repo)
    assert info["n_images"] == 2 and len(info["commit"]) == 40 and info["image_ids"] == {"ISIC_1", "ISIC_2"}
    _write(labels, [["ISIC_1", 0, 0, 0, 0], ["ISIC_2", 0, 1, 0, 0]], header)
    with pytest.raises(SystemExit, match="differs from the committed"):
        evaluate.committed_test_labels(labels, repo)
