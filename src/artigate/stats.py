"""Bootstrap confidence intervals and paired comparisons over per-image metrics."""
import numpy as np
from scipy import stats


def bootstrap_ci(values, n_boot=2000, seed=0, level=0.95):
    """Image-level nonparametric percentile bootstrap CI of the mean."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("bootstrap needs at least one value")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(n_boot, values.size))].mean(axis=1)
    tail = (1.0 - level) / 2.0 * 100.0
    return {
        "point": float(values.mean()),
        "ci_low": float(np.percentile(means, tail)),
        "ci_high": float(np.percentile(means, 100.0 - tail)),
    }


def paired_comparison(model_values, baseline_values, n_boot=2000, seed=0, level=0.95):
    """Mean paired difference (model - baseline) with bootstrap CI and Wilcoxon signed-rank p."""
    model_values = np.asarray(model_values, dtype=float)
    baseline_values = np.asarray(baseline_values, dtype=float)
    if model_values.shape != baseline_values.shape:
        raise ValueError("paired comparison needs equally sized, aligned arrays")
    difference = model_values - baseline_values
    ci = bootstrap_ci(difference, n_boot=n_boot, seed=seed, level=level)
    p_value = float(stats.wilcoxon(model_values, baseline_values).pvalue) if np.any(difference != 0) else 1.0
    return {"mean_difference": ci["point"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"], "wilcoxon_p": p_value}
