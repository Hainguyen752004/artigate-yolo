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
    assert set(ROWS) == {"A0", "A1", "A1s", "A2", "A3", "A4", "A5", "A7"}
    for spec in ROWS.values():
        assert (ROOT / spec["model"]).is_file() and (ROOT / spec["data"]).is_file()
    assert ROWS["A5"]["model"] == ROWS["A4"]["model"] and ROWS["A5"]["p_synth"] == 0
    assert ROWS["A1s"]["model"] == ROWS["A1"]["model"] and ROWS["A1s"]["data"] != ROWS["A1"]["data"]


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


def test_check_provenance_accepts_a_correct_run_and_catches_mix_ups(tmp_path, load_script):
    check = load_script("check_provenance")
    assert check.check_run(finished_run(tmp_path, "A5_s0", "A5"), COMMIT, SCHEDULE, ROWS) == []
    # an A4 run filed under the A5 name: same model YAML, but no p_synth override
    problems = check.check_run(finished_run(tmp_path, "A5_s1", "A4"), COMMIT, SCHEDULE, ROWS)
    assert any("p_synth" in problem for problem in problems)
    # A1 data used for A1s
    problems = check.check_run(finished_run(tmp_path, "A1s_s0", "A1"), COMMIT, SCHEDULE, ROWS)
    assert any(problem.startswith("data ") for problem in problems)
    dirty = finished_run(tmp_path, "A4_s0", "A4", git={"commit": COMMIT, "dirty": True})
    assert check.check_run(dirty, COMMIT, SCHEDULE, ROWS) == ["code had uncommitted changes"]
    sgd = finished_run(tmp_path, "A4_s1", "A4", hyp={**SCHEDULE, "optimizer": "auto"})
    assert any("optimizer" in problem for problem in check.check_run(sgd, COMMIT, SCHEDULE, ROWS))
    assert any("tag commit" in problem for problem in check.check_run(sgd, "c" * 40, SCHEDULE, ROWS))


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
