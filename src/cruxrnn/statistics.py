"""Paired simulation-level inference; predictions are never statistical subjects."""

import numpy as np
from scipy import stats


def holm(p_values) -> np.ndarray:
    """Holm step-down familywise adjusted p-values, returned in input order."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Expected a vector of finite p-values in [0,1]")
    order = np.argsort(values)
    adjusted = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    result = np.empty_like(values)
    result[order] = adjusted
    return result


def sign_flip(differences, rng: np.random.RandomState, samples: int = 10000) -> float:
    """Two-sided paired sign-flip test, with Monte Carlo plus-one correction.

    This assumes sign exchangeability/symmetry under the null. It is not an
    assumption-free causal randomization test: schedules were not randomly
    assigned to human participants. Identically zero differences give p=1.
    """
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or samples < 1:
        raise ValueError("At least two finite paired differences and positive samples are required")
    observed = abs(values.mean())
    if not np.any(values):
        return 1.0
    signs = rng.choice([-1, 1], size=(samples, len(values)))
    null = np.abs((signs * values).mean(axis=1))
    return float((1 + np.count_nonzero(null >= observed - 1e-12)) / (samples + 1))


def paired_summary(blocked, interleaved, rng: np.random.RandomState, samples: int = 10000) -> dict:
    """Percent/pp estimates, sample SD, paired dz, percentile CIs and two tests."""
    blocked, interleaved = np.asarray(blocked, float), np.asarray(interleaved, float)
    if blocked.ndim != 1 or blocked.shape != interleaved.shape or len(blocked) < 2:
        raise ValueError("Expected aligned vectors with at least two independent pairs")
    if not np.isfinite(blocked).all() or not np.isfinite(interleaved).all():
        raise ValueError("Missing/nonfinite pairs must be accounted for, never silently dropped")
    if samples < 1:
        raise ValueError("Bootstrap sample count must be positive")
    differences = 100 * (interleaved - blocked)
    resampled = differences[rng.randint(len(differences), size=(samples, len(differences)))]
    means, sds = resampled.mean(axis=1), resampled.std(axis=1, ddof=1)
    sd = float(differences.std(ddof=1))
    valid = sds > 0
    dz = float(differences.mean() / sd) if sd > 0 else None
    dz_interval = (
        np.percentile(means[valid] / sds[valid], [2.5, 97.5]) if valid.any() else [None, None]
    )
    if sd == 0:
        t_p = 1.0 if differences.mean() == 0 else 0.0
    else:
        t_p = float(stats.ttest_rel(interleaved, blocked).pvalue)
    return {
        "n": len(differences),
        "blocked_pct": float(100 * blocked.mean()),
        "blocked_sd_pct": float(100 * blocked.std(ddof=1)),
        "interleaved_pct": float(100 * interleaved.mean()),
        "interleaved_sd_pct": float(100 * interleaved.std(ddof=1)),
        "difference_pp": float(differences.mean()),
        "median_pp": float(np.median(differences)),
        "sd_difference_pp": sd,
        "ci_low_pp": float(np.percentile(means, 2.5)),
        "ci_high_pp": float(np.percentile(means, 97.5)),
        "paired_dz": dz,
        "dz_ci_low": float(dz_interval[0]) if dz_interval[0] is not None else None,
        "dz_ci_high": float(dz_interval[1]) if dz_interval[1] is not None else None,
        "dz_bootstrap_undefined_count": int((~valid).sum()),
        "p_sign_flip": sign_flip(differences, rng, samples),
        "p_t_sensitivity": t_p,
        "trimmed_mean_pp": float(stats.trim_mean(differences, 0.1)),
    }


def similarity_interaction(
    matrix, distances, rng: np.random.RandomState, samples: int = 10000
) -> dict:
    """Slope of IP-minus-blocked vs k and a categorical interaction test.

    Resampling and sign flips retain complete rows (one independent replicate
    ID across all distances). Categorical row-centering removes the schedule
    main effect before testing heterogeneity. The slope is a summary, not an
    assumption that the categorical response is linear.
    """
    values, x = np.asarray(matrix, float), np.asarray(distances, float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != len(x) or len(x) < 2:
        raise ValueError("Interaction requires an n-by-k matrix with n,k >=2")
    if not np.isfinite(values).all() or not np.isfinite(x).all() or len(set(x)) != len(x):
        raise ValueError("Distances must be distinct and all cells finite")
    centered = x - x.mean()
    slopes = values @ centered / (centered @ centered)
    indices = rng.randint(len(values), size=(samples, len(values)))
    resampled_slopes = slopes[indices].mean(axis=1)
    residuals = values - values.mean(axis=1, keepdims=True)
    statistic = float(np.square(residuals.mean(axis=0)).sum())
    signs = rng.choice([-1, 1], size=(samples, len(values)))
    null = np.square(signs @ residuals / len(values)).sum(axis=1)
    leave_one_out = (slopes.sum() - slopes) / (len(slopes) - 1)
    return {
        "n": len(values),
        "distances": x.tolist(),
        "slope_pp_per_hamming": float(slopes.mean()),
        "ci_low": float(np.percentile(resampled_slopes, 2.5)),
        "ci_high": float(np.percentile(resampled_slopes, 97.5)),
        "p_slope_sign_flip": sign_flip(slopes, rng, samples),
        "p_categorical_interaction": float(
            (1 + np.count_nonzero(null >= statistic - 1e-12)) / (samples + 1)
        ),
        "categorical_statistic": statistic,
        "leave_one_out_slope_min": float(leave_one_out.min()),
        "leave_one_out_slope_max": float(leave_one_out.max()),
        "bootstrap_samples": samples,
        "bootstrap_unit": "whole replicate IDs",
        "bootstrap_indices": indices,
        "bootstrap_slopes": resampled_slopes,
    }
