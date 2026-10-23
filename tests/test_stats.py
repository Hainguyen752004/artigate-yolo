import numpy as np
import pytest

from artigate.stats import bootstrap_ci, paired_comparison


def test_bootstrap_ci_brackets_the_mean_and_is_seeded():
    values = np.random.default_rng(1).uniform(0.6, 0.95, 300)
    first = bootstrap_ci(values, n_boot=500, seed=7)
    assert first["ci_low"] < first["point"] < first["ci_high"]
    assert first["point"] == pytest.approx(values.mean())
    assert bootstrap_ci(values, n_boot=500, seed=7) == first


def test_paired_comparison_detects_consistent_improvement():
    rng = np.random.default_rng(2)
    baseline = rng.uniform(0.7, 0.9, 200)
    model = baseline + 0.02 + rng.normal(0, 0.005, 200)
    result = paired_comparison(model, baseline, n_boot=500, seed=0)
    assert result["mean_difference"] == pytest.approx(0.02, abs=0.002)
    assert result["ci_low"] > 0
    assert result["wilcoxon_p"] < 1e-6


def test_paired_comparison_of_identical_inputs():
    values = np.linspace(0.5, 0.9, 50)
    result = paired_comparison(values, values, n_boot=100)
    assert result["mean_difference"] == 0.0 and result["wilcoxon_p"] == 1.0


def test_paired_comparison_requires_aligned_arrays():
    with pytest.raises(ValueError, match="aligned"):
        paired_comparison([0.1, 0.2], [0.1])


def test_holm_adjust_matches_the_step_down_definition():
    from artigate.stats import holm_adjust

    # sorted p: 0.01, 0.02, 0.03, 0.04 -> 4*0.01, max(.04, 3*.02), max(.06, 2*.03), max(.06, .04)
    assert holm_adjust([0.04, 0.01, 0.03, 0.02]) == pytest.approx([0.06, 0.04, 0.06, 0.06])
    assert holm_adjust([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    assert holm_adjust([]) == []
