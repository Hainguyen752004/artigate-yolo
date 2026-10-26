import json
from pathlib import Path
import subprocess

import pytest
import yaml

from artigate.provenance import git_state

ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = yaml.safe_load((ROOT / "configs" / "train" / "default.yaml").read_text(encoding="utf-8"))
ROWS = yaml.safe_load((ROOT / "configs" / "runs.yaml").read_text(encoding="utf-8"))["rows"]
COMMIT = "a" * 40


def test_every_row_points_at_existing_configs():
    assert set(ROWS) == {"A0", "A1", "A1s", "A2", "A3", "A4", "A5", "A7", "Y26", "A1v7", "A1v17"}
    for spec in ROWS.values():
        assert (ROOT / spec["model"]).is_file() and (ROOT / spec["data"]).is_file()
    assert ROWS["A5"]["model"] == ROWS["A4"]["model"] and ROWS["A5"]["p_synth"] == 0
    assert ROWS["A1s"]["model"] == ROWS["A1"]["model"] and ROWS["A1s"]["data"] != ROWS["A1"]["data"]
    assert ROWS["A1v7"]["views"] == {"train": ["v7"], "eval": ["v7"]}
    assert ROWS["A1v17"]["views"] == {"train": ["v1", "v7"], "eval": ["v1"]}


def test_plan_names_runs_and_loops_over_seeds_first(load_script):
    run_plan = load_script("run_plan")
    runs = run_plan.plan(["A4", "A5"], [1, 2], ROWS)
    assert [name for name, _ in runs] == ["A4_s1", "A5_s1", "A4_s2", "A5_s2"]
    assert runs[1][1][-2:] == ["--p-synth", "0"] and "--p-synth" not in runs[0][1]
    assert runs[0][1][runs[0][1].index("--name") + 1] == "A4_s1"
    with pytest.raises(SystemExit, match="unknown rows"):
        run_plan.plan(["A6"], [0], ROWS)


def test_dry_run_skips_finished_runs_and_stops_at_interrupted_ones(tmp_path, load_script, capsys):
    run_plan = load_script("run_plan")
    (tmp_path / "A1_s0").mkdir()
    (tmp_path / "A1_s0" / "provenance.json").write_text("{}", encoding="utf-8")
    run_plan.main(["--rows", "A1", "A1s", "--seeds", "0", "--project", str(tmp_path), "--dry-run"])
    output = capsys.readouterr().out
    assert "A1_s0: finished, skipped" in output and "A1s_s0: --model" in output
    (tmp_path / "A1s_s0").mkdir()
    with pytest.raises(SystemExit, match="interrupted run"):
        run_plan.main(["--rows", "A1s", "--seeds", "0", "--project", str(tmp_path), "--dry-run"])


def test_training_requires_the_tagged_clean_commit(monkeypatch, load_script):
    run_plan = load_script("run_plan")
    monkeypatch.setattr(run_plan, "tag_commit", lambda tag, root=None: COMMIT)
    monkeypatch.setattr(run_plan, "git_state", lambda root: {"commit": "b" * 40, "dirty": False})
    with pytest.raises(SystemExit, match="git checkout train-v1"):
        run_plan.check_code_version("train-v1")
    monkeypatch.setattr(run_plan, "git_state", lambda root: {"commit": COMMIT, "dirty": True})
    with pytest.raises(SystemExit, match="uncommitted"):
        run_plan.check_code_version("train-v1")
    monkeypatch.setattr(run_plan, "git_state", lambda root: {"commit": COMMIT, "dirty": False})
    assert run_plan.check_code_version("train-v1") == COMMIT


def test_git_state_ignores_untracked_notes_but_not_untracked_code(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "test")
    (tmp_path / "README.md").write_text("x", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-q", "-m", "init")
    (tmp_path / "note.txt").write_text("personal note", encoding="utf-8")
    assert git_state(tmp_path)["dirty"] is False and git_state(tmp_path)["untracked"] == ["note.txt"]
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "extra.py").write_text("x = 1", encoding="utf-8")
    assert git_state(tmp_path)["dirty"] is True
    (tmp_path / "src" / "extra.py").unlink()
    (tmp_path / "README.md").write_text("changed", encoding="utf-8")
    assert git_state(tmp_path)["dirty"] is True


def finished_run(root, name, row, **changes):
    spec = ROWS[row]
    provenance = {
        "model_yaml": str(ROOT / spec["model"]), "data_yaml": str(ROOT / spec["data"]), "seed": int(name.split("_s")[1]),
        "hyp": dict(SCHEDULE), "git": {"commit": COMMIT, "dirty": False},
        "artigate": {"data_overrides": {"p_synth": spec["p_synth"]} if "p_synth" in spec else {}},
    }
    for key, value in changes.items():
        provenance[key] = value
    run_dir = root / name
    run_dir.mkdir()
    (run_dir / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    args = {key: SCHEDULE[key] for key in ("optimizer", "lr0", "epochs", "batch")}
    (run_dir / "args.yaml").write_text(yaml.safe_dump(args), encoding="utf-8")
    return run_dir


EXPORT_VIEWS = {"isic2018": {"train": ["v1"], "eval": ["v1"]}, "isic2018_art": {"train": ["v1"], "eval": ["v1"]},
                "isic2018_v7": {"train": ["v7"], "eval": ["v7"]}, "isic2018_v17": {"train": ["v1", "v7"], "eval": ["v1"]}}


def fake_views(data_yaml):
    return EXPORT_VIEWS[Path(data_yaml).stem]


def test_check_provenance_accepts_a_correct_run_and_catches_mix_ups(tmp_path, load_script):
    check = load_script("check_provenance")

    def problems_of(run_dir, commit=COMMIT, views_of=fake_views):
        return check.check_run(run_dir, commit, SCHEDULE, ROWS, views_of=views_of)

    assert problems_of(finished_run(tmp_path, "A5_s0", "A5")) == []
    assert problems_of(finished_run(tmp_path, "A1v17_s0", "A1v17")) == []
    # an A4 run filed under the A5 name: same model YAML, but no p_synth override
    assert any("p_synth" in problem for problem in problems_of(finished_run(tmp_path, "A5_s1", "A4")))
    # A1 data used for A1s, and an A1 run filed as A1v7
    assert any(problem.startswith("data ") for problem in problems_of(finished_run(tmp_path, "A1s_s0", "A1")))
    assert any(problem.startswith("data ") for problem in problems_of(finished_run(tmp_path, "A1v7_s0", "A1")))
    # the A1v17 data YAML pointing at an export made without the v7 training view
    v1_only = problems_of(finished_run(tmp_path, "A1v17_s1", "A1v17"), views_of=lambda path: EXPORT_VIEWS["isic2018"])
    assert any("views" in problem for problem in v1_only)
    dirty = finished_run(tmp_path, "A4_s0", "A4", git={"commit": COMMIT, "dirty": True})
    assert problems_of(dirty) == ["code had uncommitted changes"]
    sgd = finished_run(tmp_path, "A4_s1", "A4", hyp={**SCHEDULE, "optimizer": "auto"})
    assert any("optimizer" in problem for problem in problems_of(sgd))
    assert any("tag commit" in problem for problem in problems_of(sgd, commit="c" * 40))


def test_y26_maps_every_stock_layer_and_p2_rows_only_the_prefix():
    from artigate.transfer import stock_layer_map

    assert stock_layer_map(ROOT / ROWS["Y26"]["model"]) == {index: index for index in range(24)}
    assert len(stock_layer_map(ROOT / ROWS["A0"]["model"])) == 17
    assert len(stock_layer_map(ROOT / ROWS["A4"]["model"])) == 17


@pytest.mark.skipif(not (ROOT / "weights" / "yolo26n-seg.pt").is_file(), reason="weights/yolo26n-seg.pt is not present")
def test_y26_receives_every_stock_tensor_except_the_class_dependent_ones():
    from artigate.artifact_loss import ArtiGateSegModel
    from artigate.modules import register_modules
    from artigate.transfer import load_partial_pretrained, stock_layer_map

    register_modules()
    model = ArtiGateSegModel(str(ROOT / ROWS["Y26"]["model"]), nc=1, ch=3, verbose=False)
    report = load_partial_pretrained(model, ROOT / "weights" / "yolo26n-seg.pt", stock_layer_map(ROOT / ROWS["Y26"]["model"]))
    loaded = {entry["destination"] for entry in report["mapped"]}
    missing = [key for key in model.state_dict() if key not in loaded]
    assert not report["missing_destination"] and len(loaded) == 740
    # what Ultralytics also skips when fine-tuning to one class: the nc-wide class branches
    assert all(key.startswith(("model.23.cv3.", "model.23.one2one_cv3.", "model.23.proto.semseg.2.")) for key in missing)


def test_inspect_run_reads_beta_of_every_aga_block(load_script):
    from ultralytics import YOLO

    from artigate.modules import register_modules

    register_modules()
    model = YOLO(str(ROOT / ROWS["A4"]["model"])).model
    values = load_script("inspect_run").gate_values(model)
    assert len(values) == 2 and all(beta == 0.0 and weight == 0.0 for _, beta, weight in values)
    first = next(module for module in model.modules() if type(module).__name__ == "AGA")
    first.beta.data.fill_(0.3)
    assert load_script("inspect_run").gate_values(model)[0][1] == pytest.approx(0.3)
