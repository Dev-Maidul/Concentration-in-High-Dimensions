"""
dependence_analysis.py
-----------------------
Empirical analysis supporting the heuristic derivation of the -0.623
distance contrast exponent (Section 5.2 of the paper).

Key question: Why does Crel(d) ~ d^{-0.623} rather than d^{-0.5}?

The naive theory assuming independence of all O(n^2) pairwise distances
predicts d^{-0.5}. The actual exponent is steeper because pairwise
distances sharing a common point are positively correlated.

This module:
  1. Analytically computes Corr(Dij^2, Dik^2) = 0.5 for Gaussian vectors
  2. Empirically verifies this correlation across dimensions
  3. Estimates the effective number of independent distances n_eff
  4. Shows how n_eff modifies the extreme value scaling
  5. Tests whether the corrected exponent matches -0.623 empirically
  6. Verifies the exponent across multiple n values (robustness check)
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Analytical correlation Corr(Dij^2, Dik^2)
# ==============================================================================

def analytical_correlation() -> Dict:
    """
    Derive analytically that Corr(Dij^2, Dik^2) = 0.5 for Gaussian vectors.

    For Xi, Xj, Xk ~ N(0, Id) independently:
      Dij^2 = ||Xi - Xj||^2 = sum_l (Xil - Xjl)^2
      Dik^2 = ||Xi - Xk||^2 = sum_l (Xil - Xkl)^2

    Each coordinate difference Uil = Xil - Xjl ~ N(0,2)
    Each coordinate difference Vil = Xil - Xkl ~ N(0,2)

    Cov(Uil^2, Vil^2) = E[Uil^2 * Vil^2] - E[Uil^2] * E[Vil^2]

    Since Uil = Xil - Xjl and Vil = Xil - Xkl share Xil:
      Cov(Uil, Vil) = Var(Xil) = 1
      Corr(Uil, Vil) = 1/2  (since both have variance 2)

    For correlated bivariate normal (U,V) with correlation rho:
      E[U^2 V^2] = 1 + 2*rho^2  (when both standardised)
      Cov(U^2, V^2) = 2*rho^2

    Scaling back: Var(Uil^2) = 2*(Var(Uil))^2 = 2*4 = 8
    Cov(Uil^2, Vil^2) = 2*(Cov(Uil,Vil))^2 = 2*1 = 2

    Summing over d independent coordinates:
      Cov(Dij^2, Dik^2) = d * 2
      Var(Dij^2) = d * 8
      Corr(Dij^2, Dik^2) = d*2 / d*8 = 1/4

    Note: The exact value is 1/4, not 1/2 as for the squared norms.
    This module verifies this analytically and empirically.

    Returns
    -------
    dict with analytical result and derivation details
    """
    result = {
        "analytical_correlation_D2": 0.25,  # Corr(Dij^2, Dik^2) = 1/4
        "derivation": {
            "Cov_Uil2_Vil2": 2.0,          # per coordinate
            "Var_Dij2_per_coord": 8.0,
            "corr_formula": "d*2 / (d*8) = 1/4",
        },
        "implication": (
            "Corr(Dij^2, Dik^2) = 1/4 means pairwise distances sharing a "
            "point are positively correlated. This reduces the effective number "
            "of independent distances below n*(n-1)/2, shifting the extreme "
            "value scaling exponent from -0.5 toward -0.623."
        )
    }
    logger.info("Analytical Corr(Dij^2, Dik^2) = 1/4")
    return result


# ==============================================================================
# 2. Empirical correlation verification
# ==============================================================================

def empirical_correlation(d: int, n_points: int = 500,
                           n_trials: int = 100,
                           seed: int = 42) -> Dict:
    """
    Empirically measure Corr(Dij^2, Dik^2) for Gaussian vectors.

    For each trial, samples n_points vectors in R^d and computes
    the correlation between squared distances that share a point.

    Parameters
    ----------
    d        : dimension
    n_points : points per trial (enough pairs to estimate correlation)
    n_trials : independent trials
    seed     : reproducibility

    Returns
    -------
    dict with mean_corr, std_corr, analytical_value
    """
    rng = np.random.default_rng(seed)
    correlations = []

    for t in range(n_trials):
        X = rng.standard_normal((n_points, d))

        # For each point i, compute distances to all j != i and k != i, k != j
        # Sample n_pairs triples (i, j, k) with i != j != k
        n_pairs = min(1000, n_points * (n_points - 1) // 2)
        i_idx = rng.integers(0, n_points, size=n_pairs)
        j_idx = rng.integers(0, n_points, size=n_pairs)
        k_idx = rng.integers(0, n_points, size=n_pairs)

        # Ensure all different
        valid = (i_idx != j_idx) & (i_idx != k_idx) & (j_idx != k_idx)
        i_v, j_v, k_v = i_idx[valid], j_idx[valid], k_idx[valid]

        if len(i_v) < 10:
            continue

        # Compute squared distances
        dij2 = np.sum((X[i_v] - X[j_v]) ** 2, axis=1)
        dik2 = np.sum((X[i_v] - X[k_v]) ** 2, axis=1)

        # Pearson correlation
        corr = np.corrcoef(dij2, dik2)[0, 1]
        if np.isfinite(corr):
            correlations.append(corr)

    correlations = np.array(correlations)
    result = {
        "d":                  d,
        "mean_corr":          float(np.mean(correlations)),
        "std_corr":           float(np.std(correlations)),
        "analytical_value":   0.25,
        "n_trials":           len(correlations),
    }
    logger.info(
        "d=%d  Corr(Dij^2,Dik^2) = %.4f ± %.4f  (analytical=0.25)",
        d, result["mean_corr"], result["std_corr"]
    )
    return result


def run_correlation_experiment(dimensions: List[int] = None,
                                n_trials: int = 100,
                                seed: int = 42) -> Dict:
    """
    Run empirical correlation verification across multiple dimensions.

    Tests whether Corr(Dij^2, Dik^2) = 1/4 is dimension-independent.

    Returns
    -------
    dict mapping dimension to correlation results
    """
    if dimensions is None:
        dimensions = [5, 10, 20, 50, 100, 200, 500]

    logger.info("Running correlation experiment across %d dimensions …", len(dimensions))
    analytical = analytical_correlation()
    results = {"analytical": analytical, "empirical": {}}

    for d in dimensions:
        logger.info("  d = %d", d)
        res = empirical_correlation(d, n_trials=n_trials, seed=seed)
        results["empirical"][d] = res

    return results


# ==============================================================================
# 3. Effective number of independent distances
# ==============================================================================

def effective_independent_distances(n: int, rho: float = 0.25) -> Dict:
    """
    Compute the effective number of independent distances given
    pairwise correlation rho among distances sharing a point.

    Model: n points, n*(n-1)/2 total pairs.
    Each pair (i,j) correlates with pairs (i,k) and (k,j) for all k.
    Each point participates in (n-1) distances.
    Total correlated pairs per distance: 2*(n-2) (two endpoints).

    Following the effective sample size formula for correlated samples:
      n_eff = n_total / (1 + (mean_corr_pairs) * rho)

    For the extreme value problem (max and min), the relevant scaling
    involves the effective count n_eff in place of n_total.

    Parameters
    ----------
    n   : number of points
    rho : pairwise correlation between shared-endpoint distances

    Returns
    -------
    dict with n_total, n_eff, ratio, implied_exponent_shift
    """
    n_total = n * (n - 1) // 2

    # Average number of correlated partners per distance
    # Each distance (i,j) shares i with (n-2) other distances,
    # and shares j with (n-2) other distances
    # Total correlated partners = 2*(n-2)
    mean_corr_partners = 2 * (n - 2)

    # Effective sample size (Kish 1965 formula for cluster sampling)
    deff = 1 + mean_corr_partners * rho
    n_eff = n_total / deff

    # Under independence, extreme value scaling goes as log(n_total)
    # Under dependence, it scales as log(n_eff)
    # The modified Crel ~ sqrt(log(n_eff) / d)
    # Effective exponent in d: -0.5 * (1 - log(log(n_eff)) / log(d))
    # For typical n=10000, d=100..2000:
    ratio = n_eff / n_total
    log_n_total = np.log(n_total)
    log_n_eff   = np.log(max(n_eff, 1))

    # Predicted modified exponent (heuristic)
    # Exponent = -1/2 * log(n_eff) / log(n_total) * scaling_adjustment
    # Numerically estimated from the log ratio
    exponent_shift = 0.5 * (1 - log_n_eff / log_n_total)
    predicted_exponent = -(0.5 + exponent_shift)

    result = {
        "n":                     n,
        "n_total":               n_total,
        "n_eff":                 n_eff,
        "ratio":                 ratio,
        "rho":                   rho,
        "mean_corr_partners":    mean_corr_partners,
        "predicted_exponent":    predicted_exponent,
        "target_exponent":       -0.623,
        "exponent_gap":          predicted_exponent - (-0.623),
    }
    logger.info(
        "n=%d  n_eff=%.0f  ratio=%.3f  predicted_exponent=%.3f",
        n, n_eff, ratio, predicted_exponent
    )
    return result


# ==============================================================================
# 4. Exponent vs sample size test
# ==============================================================================

def exponent_vs_n(n_values: List[int] = None,
                   dimensions: List[int] = None,
                   n_trials: int = 30,
                   seed: int = 42) -> Dict:
    """
    Test whether the distance contrast exponent depends on n in the
    way predicted by the effective-independence heuristic.

    If our model is correct, the exponent should vary with n as:
      β(n) ≈ -0.5 * log(n_eff(n)) / log(n_total(n))

    Parameters
    ----------
    n_values   : sample sizes to test
    dimensions : dimensions to test (for high-dimensional regime d >= 50)
    n_trials   : independent trials per (n, d) combination
    seed       : reproducibility

    Returns
    -------
    dict with exponents per n value and comparison to predictions
    """
    if n_values is None:
        n_values = [500, 1000, 5000, 10000, 50000]
    if dimensions is None:
        dimensions = [50, 100, 200, 500, 1000]

    rng = np.random.default_rng(seed)
    results = {"n_values": n_values, "dimensions": dimensions, "data": {}}

    for n in n_values:
        logger.info("n = %d", n)
        results["data"][n] = {}
        contrasts_by_d = {d: [] for d in dimensions}

        for trial in range(n_trials):
            trial_seed = seed + trial * 10000 + n
            np.random.seed(trial_seed)
            for d in dimensions:
                X = np.random.randn(n, d)
                # Subsample for efficiency
                sub = min(500, n)
                idx = np.random.choice(n, sub, replace=False)
                Xs  = X[idx]

                # Compute pairwise distances
                diff = Xs[:, None, :] - Xs[None, :, :]
                dmat = np.sqrt((diff ** 2).sum(axis=2))
                upper_idx = np.triu_indices(sub, k=1)
                dists = dmat[upper_idx]

                d_min = dists.min()
                d_max = dists.max()
                if d_min > 1e-10:
                    crel = (d_max - d_min) / d_min
                    contrasts_by_d[d].append(crel)

        # Fit power law for each n
        from scipy import stats as sp_stats
        fitted_exponents = {}
        for d in dimensions:
            c_arr = np.array(contrasts_by_d[d])
            if len(c_arr) > 0:
                fitted_exponents[d] = float(np.mean(c_arr))

        # Fit exponent across d values (for d >= 50)
        d_arr = np.array([d for d in dimensions if d >= 50])
        c_arr = np.array([np.mean(contrasts_by_d[d]) for d in d_arr])
        valid = (d_arr > 0) & (c_arr > 0) & np.isfinite(c_arr)

        if valid.sum() >= 3:
            slope, intercept, r, p, se = sp_stats.linregress(
                np.log(d_arr[valid]), np.log(c_arr[valid])
            )
            fitted_exp = slope
        else:
            fitted_exp = None

        # Predicted exponent from our model
        predicted = effective_independent_distances(n, rho=0.25)

        results["data"][n] = {
            "contrasts_by_d":    {d: float(np.mean(v)) for d, v in contrasts_by_d.items()},
            "fitted_exponent":   fitted_exp,
            "predicted_exponent": predicted["predicted_exponent"],
            "n_eff":             predicted["n_eff"],
        }
        logger.info(
            "  n=%d  fitted_exponent=%.3f  predicted=%.3f",
            n, fitted_exp or -999, predicted["predicted_exponent"]
        )

    return results


# ==============================================================================
# 5. Universality test across distributions
# ==============================================================================

def universality_test(distributions: List[str] = None,
                       dimensions: List[int] = None,
                       n_samples: int = 10_000,
                       n_trials: int = 50,
                       seed: int = 42) -> Dict:
    """
    Test universality of the -0.623 exponent across many distributions.

    Extends the 4 distributions in the paper to 8-10 distributions
    spanning different tail behaviors.

    Returns
    -------
    dict with fitted exponents and confidence intervals per distribution
    """
    if distributions is None:
        distributions = [
            "gaussian", "uniform", "laplace", "student_t3",
            "student_t5", "student_t10", "exponential", "beta22",
            "mixed_gaussian", "cauchy_truncated"
        ]
    if dimensions is None:
        dimensions = [50, 100, 200, 500, 1000, 2000]

    from scipy import stats as sp_stats
    rng_global = np.random.default_rng(seed)
    results = {
        "distributions": distributions,
        "dimensions": dimensions,
        "n_samples": n_samples,
        "n_trials": n_trials,
        "data": {}
    }

    def sample_distribution(dist_name: str, n: int, d: int,
                              rng: np.random.Generator) -> np.ndarray:
        if dist_name == "gaussian":
            return rng.standard_normal((n, d))
        elif dist_name == "uniform":
            return rng.uniform(-np.sqrt(3), np.sqrt(3), (n, d))
        elif dist_name == "laplace":
            return rng.laplace(0, 1 / np.sqrt(2), (n, d))
        elif dist_name == "student_t3":
            return rng.standard_t(3, (n, d)) / np.sqrt(3)
        elif dist_name == "student_t5":
            return rng.standard_t(5, (n, d)) / np.sqrt(5 / 3)
        elif dist_name == "student_t10":
            return rng.standard_t(10, (n, d)) / np.sqrt(10 / 8)
        elif dist_name == "exponential":
            return (rng.exponential(1, (n, d)) - 1.0)  # mean=0, var=1
        elif dist_name == "beta22":
            raw = rng.beta(2, 2, (n, d))  # mean=0.5, var=1/20
            return (raw - 0.5) / np.sqrt(1 / 20)  # standardise
        elif dist_name == "mixed_gaussian":
            mask = rng.random((n, d)) < 0.7
            narrow = rng.standard_normal((n, d))
            wide   = rng.standard_normal((n, d)) * 2
            raw    = np.where(mask, narrow, wide)
            # Variance = 0.7*1 + 0.3*4 = 1.9, standardise
            return raw / np.sqrt(1.9)
        elif dist_name == "cauchy_truncated":
            # Truncated Cauchy at ±10 (finite moments)
            raw = rng.standard_cauchy((n, d))
            raw = np.clip(raw, -10, 10)
            raw = raw / np.std(raw.ravel())  # standardise
            return raw
        else:
            raise ValueError(f"Unknown distribution: {dist_name}")

    for dist in distributions:
        logger.info("Distribution: %s", dist)
        contrasts = {d: [] for d in dimensions}

        for trial in range(n_trials):
            t_seed = seed + trial
            rng    = np.random.default_rng(t_seed)

            for d in dimensions:
                try:
                    X = sample_distribution(dist, n_samples, d, rng)
                    # Subsample for speed
                    sub = 500
                    idx = rng.choice(n_samples, sub, replace=False)
                    Xs  = X[idx]

                    diff = Xs[:, None, :] - Xs[None, :, :]
                    dmat = np.sqrt((diff ** 2).sum(axis=2))
                    upper = dmat[np.triu_indices(sub, k=1)]

                    d_min = upper.min()
                    if d_min > 1e-10:
                        crel = (upper.max() - d_min) / d_min
                        contrasts[d].append(crel)
                except Exception as e:
                    logger.warning("  d=%d trial=%d error: %s", d, trial, e)

        # Fit exponent for d >= 50
        d_arr = np.array([d for d in dimensions if d >= 50])
        c_arr = np.array([np.mean(contrasts[d]) if contrasts[d] else np.nan
                          for d in d_arr])
        valid = np.isfinite(c_arr) & (c_arr > 0) & (d_arr > 0)

        if valid.sum() >= 3:
            slope, intercept, r, p, se = sp_stats.linregress(
                np.log(d_arr[valid]), np.log(c_arr[valid])
            )
            # Bootstrap confidence interval on exponent
            boot_slopes = []
            log_d = np.log(d_arr[valid])
            for _ in range(1000):
                boot_idx = np.random.choice(valid.sum(), valid.sum(), replace=True)
                bs, bi, _, _, _ = sp_stats.linregress(
                    log_d[boot_idx], np.log(c_arr[valid])[boot_idx]
                )
                boot_slopes.append(bs)
            ci_low  = float(np.percentile(boot_slopes, 2.5))
            ci_high = float(np.percentile(boot_slopes, 97.5))
        else:
            slope, se, ci_low, ci_high, r = None, None, None, None, None

        results["data"][dist] = {
            "contrasts_mean": {d: float(np.mean(contrasts[d])) if contrasts[d] else None
                               for d in dimensions},
            "contrasts_std":  {d: float(np.std(contrasts[d]))  if contrasts[d] else None
                               for d in dimensions},
            "fitted_exponent":      slope,
            "std_error":            se,
            "ci_95_low":            ci_low,
            "ci_95_high":           ci_high,
            "r_squared":            r ** 2 if r is not None else None,
        }
        logger.info("  exponent = %.4f ± %.4f  [%.4f, %.4f]",
                    slope or -999, se or -999, ci_low or -999, ci_high or -999)

    # Summary statistics
    exponents = [v["fitted_exponent"] for v in results["data"].values()
                 if v["fitted_exponent"] is not None]
    results["summary"] = {
        "mean_exponent":   float(np.mean(exponents)),
        "std_exponent":    float(np.std(exponents)),
        "min_exponent":    float(np.min(exponents)),
        "max_exponent":    float(np.max(exponents)),
        "target":          -0.623,
        "range_pct":       float(100 * (np.max(exponents) - np.min(exponents))
                                 / abs(np.mean(exponents))),
        # t-test against null hypothesis of -0.5
        "t_test_vs_minus05": sp_stats.ttest_1samp(exponents, -0.5).pvalue,
    }
    logger.info(
        "Universality: mean_exp=%.4f ± %.4f  range=%.1f%%  t-test p=%.4e",
        results["summary"]["mean_exponent"],
        results["summary"]["std_exponent"],
        results["summary"]["range_pct"],
        results["summary"]["t_test_vs_minus05"],
    )

    return results
