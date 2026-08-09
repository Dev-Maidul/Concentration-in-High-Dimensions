"""
intrinsic_dim.py
-----------------
Intrinsic dimensionality estimation using multiple complementary methods.

Methods implemented:
  1. TwoNN MLE      (Facco et al. 2017)
  2. MLE estimator  (Levina & Bickel 2005)
  3. PCA d_95       (number of PCs for 95% variance)
  4. Participation Ratio  (from eigenvalue spectrum)
  5. Correlation Dimension (box-counting slope)

For the paper, we report TwoNN and MLE as the primary pair, with PCA
as a sanity check. If TwoNN and MLE agree within 20% we use their
average as d_int. If they disagree, we report both and note it.
"""

import logging
import numpy as np
from typing import Dict, Optional, Tuple
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. TwoNN MLE estimator  (Facco et al. 2017)
# ==============================================================================

def twonn(X: np.ndarray,
          subsample: int = 5_000,
          seed: int = 42) -> Dict:
    """
    Two-Nearest-Neighbour MLE intrinsic dimension estimator.

    For each point i, let r1 = distance to 1st NN, r2 = distance to 2nd NN.
    The ratio μ_i = r2/r1 follows F(μ) = 1 − μ^{−d_int}.
    MLE gives d̂ = n / Σ log(μ_i).

    Parameters
    ----------
    X         : np.ndarray (N, D)
    subsample : max points to use (chunked to avoid OOM)
    seed      : reproducibility

    Returns
    -------
    dict with keys: d_twonn, n_valid, mu_values
    """
    rng = np.random.default_rng(seed)
    n   = min(subsample, X.shape[0])
    idx = rng.choice(X.shape[0], size=n, replace=False)
    Xs  = X[idx].astype(np.float64)

    chunk_size = 500
    mu_vals = []

    for start in range(0, n, chunk_size):
        end  = min(start + chunk_size, n)
        dmat = cdist(Xs[start:end], Xs, metric="euclidean")

        # Zero out self-distances
        for local_i in range(end - start):
            dmat[local_i, start + local_i] = np.inf

        # Get two smallest distances
        part = np.partition(dmat, 2, axis=1)[:, :2]
        r1 = part[:, 0]
        r2 = part[:, 1]

        valid = r1 > 1e-12
        with np.errstate(divide="ignore", invalid="ignore"):
            mu = np.where(valid, r2 / r1, np.nan)

        mu_clean = mu[np.isfinite(mu) & (mu > 1.0)]
        mu_vals.extend(mu_clean.tolist())

    mu = np.array(mu_vals)
    if len(mu) < 10:
        logger.warning("TwoNN: too few valid μ values (%d)", len(mu))
        return {"d_twonn": None, "n_valid": len(mu), "mu_values": mu}

    d_hat = float(len(mu) / np.log(mu).sum())
    logger.info("TwoNN: d̂ = %.2f  (n_valid=%d)", d_hat, len(mu))
    return {"d_twonn": d_hat, "n_valid": len(mu), "mu_values": mu}


# ==============================================================================
# 2. Levina-Bickel MLE estimator  (2005)
# ==============================================================================

def mle_levina_bickel(X: np.ndarray,
                       k: int = 10,
                       subsample: int = 5_000,
                       seed: int = 42) -> Dict:
    """
    Levina-Bickel MLE intrinsic dimension estimator.

    For each point x, let T_k(x) = distance to k-th NN.
    d̂_k(x) = [ (1/k) Σ_{j=1}^{k} log(T_k(x)/T_j(x)) ]^{-1}
    d̂ = mean over all points.

    Parameters
    ----------
    X         : np.ndarray (N, D)
    k         : number of nearest neighbours
    subsample : max points
    seed      : reproducibility

    Returns
    -------
    dict with keys: d_mle, d_mle_std, k
    """
    rng = np.random.default_rng(seed)
    n   = min(subsample, X.shape[0])
    idx = rng.choice(X.shape[0], size=n, replace=False)
    Xs  = X[idx].astype(np.float64)

    chunk_size = 500
    d_estimates = []

    for start in range(0, n, chunk_size):
        end  = min(start + chunk_size, n)
        dmat = cdist(Xs[start:end], Xs, metric="euclidean")

        for local_i in range(end - start):
            dmat[local_i, start + local_i] = np.inf

        # Sort distances for each point
        sorted_dists = np.sort(dmat, axis=1)[:, :k+1]  # (batch, k+1)
        t_k = sorted_dists[:, k]    # k-th NN distance
        t_j = sorted_dists[:, :k]   # 1..k-1 NN distances

        # Avoid log(0)
        valid_mask = (t_k > 1e-12) & np.all(t_j > 1e-12, axis=1)
        t_k_v = t_k[valid_mask]
        t_j_v = t_j[valid_mask]

        if len(t_k_v) == 0:
            continue

        log_ratios = np.log(t_k_v[:, None]) - np.log(t_j_v)  # (m, k)
        mean_log   = log_ratios.mean(axis=1)                   # (m,)
        d_i = np.where(mean_log > 1e-12, 1.0 / mean_log, np.nan)
        d_estimates.extend(d_i[np.isfinite(d_i)].tolist())

    if not d_estimates:
        return {"d_mle": None, "d_mle_std": None, "k": k}

    d_estimates = np.array(d_estimates)
    d_hat = float(np.mean(d_estimates))
    d_std = float(np.std(d_estimates))
    logger.info("MLE (Levina-Bickel): d̂ = %.2f ± %.2f  (k=%d)", d_hat, d_std, k)
    return {"d_mle": d_hat, "d_mle_std": d_std, "k": k}


# ==============================================================================
# 3. PCA effective dimension
# ==============================================================================

def pca_effective_dim(X: np.ndarray,
                       threshold: float = 0.95,
                       subsample: int = 10_000,
                       seed: int = 42) -> Dict:
    """
    PCA-based effective dimension: number of components to explain
    `threshold` fraction of variance.

    Also computes participation ratio PR = (Σλ)² / Σλ².

    Returns
    -------
    dict with keys: d_pca, participation_ratio, explained_variance_ratios,
                    cumulative_variance
    """
    rng = np.random.default_rng(seed)
    n   = min(subsample, X.shape[0])
    idx = rng.choice(X.shape[0], size=n, replace=False)
    Xs  = X[idx].astype(np.float64)
    Xs -= Xs.mean(axis=0)

    _, s, _ = np.linalg.svd(Xs, full_matrices=False)
    eigenvalues = s ** 2
    total       = eigenvalues.sum()

    ratios     = eigenvalues / total
    cumulative = np.cumsum(ratios)

    d_pca = int(np.searchsorted(cumulative, threshold)) + 1
    pr    = float(total ** 2 / (eigenvalues ** 2).sum())

    logger.info("PCA d_95=%d  PR=%.1f  (subsample n=%d)", d_pca, pr, n)
    return {
        "d_pca":                    d_pca,
        "participation_ratio":       pr,
        "explained_variance_ratios": ratios[:50].tolist(),
        "cumulative_variance":        cumulative[:50].tolist(),
    }


# ==============================================================================
# 4. Correlation dimension
# ==============================================================================

def correlation_dim(X: np.ndarray,
                     n_pairs: int = 50_000,
                     seed: int = 42) -> Dict:
    """
    Correlation dimension estimate via log C(r) vs log r slope.

    Returns
    -------
    dict with keys: d_corr, r2
    """
    rng = np.random.default_rng(seed)
    n   = X.shape[0]

    i_idx = rng.integers(0, n, size=n_pairs)
    j_idx = rng.integers(0, n, size=n_pairs)
    mask  = i_idx != j_idx
    dists = np.linalg.norm(
        X[i_idx[mask]].astype(np.float64) - X[j_idx[mask]].astype(np.float64),
        axis=1
    )

    dists_sorted = np.sort(dists)
    lo, hi = np.percentile(dists_sorted, [5, 85])
    r_vals = np.linspace(lo, hi, 30)
    c_vals = np.array([np.mean(dists_sorted <= r) for r in r_vals])

    valid = (r_vals > 0) & (c_vals > 0)
    if valid.sum() < 5:
        return {"d_corr": None, "r2": None}

    log_r = np.log(r_vals[valid])
    log_c = np.log(c_vals[valid])
    coeffs    = np.polyfit(log_r, log_c, 1)
    residuals = log_c - np.polyval(coeffs, log_r)
    ss_res = (residuals ** 2).sum()
    ss_tot = ((log_c - log_c.mean()) ** 2).sum()
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    logger.info("Corr dim: d̂ = %.2f  R²=%.3f", float(coeffs[0]), r2)
    return {"d_corr": float(coeffs[0]), "r2": r2}


# ==============================================================================
# Unified estimator
# ==============================================================================

def estimate_intrinsic_dim(X: np.ndarray,
                            subsample: int = 5_000,
                            seed: int = 42,
                            run_corr_dim: bool = True) -> Dict:
    """
    Run all intrinsic dimension estimators and return a unified summary.

    The consensus d_int is the median of TwoNN, MLE, and PCA values.
    If TwoNN and MLE agree within 25% their average is flagged as reliable.

    Parameters
    ----------
    X             : np.ndarray (N, D)
    subsample     : points to use for expensive estimators
    seed          : reproducibility
    run_corr_dim  : whether to run correlation dimension (slow for large D)

    Returns
    -------
    dict with all estimator results plus consensus d_int
    """
    logger.info("Intrinsic dim estimation: shape=%s", X.shape)

    results = {}

    # TwoNN
    logger.info("  Running TwoNN …")
    results["twonn"] = twonn(X, subsample=subsample, seed=seed)

    # MLE Levina-Bickel
    logger.info("  Running MLE (Levina-Bickel) …")
    results["mle"] = mle_levina_bickel(X, subsample=subsample, seed=seed)

    # PCA
    logger.info("  Running PCA effective dim …")
    results["pca"] = pca_effective_dim(X, seed=seed)

    # Correlation dimension (optional)
    if run_corr_dim and X.shape[1] <= 2000:
        logger.info("  Running correlation dimension …")
        results["corr_dim"] = correlation_dim(X, seed=seed)
    else:
        results["corr_dim"] = {"d_corr": None, "r2": None}

    # Consensus
    candidates = []
    d_twonn = results["twonn"].get("d_twonn")
    d_mle   = results["mle"].get("d_mle")
    d_pca   = results["pca"].get("d_pca")

    for v in [d_twonn, d_mle, d_pca]:
        if v is not None and not np.isnan(v) and v > 0:
            candidates.append(v)

    consensus = float(np.median(candidates)) if candidates else None

    # Agreement check between TwoNN and MLE
    if d_twonn is not None and d_mle is not None:
        ratio = max(d_twonn, d_mle) / max(min(d_twonn, d_mle), 1e-6)
        agreement = ratio <= 1.25  # within 25%
    else:
        agreement = False

    results["consensus"] = {
        "d_int":            consensus,
        "twonn_mle_agree":  agreement,
        "d_twonn":          d_twonn,
        "d_mle":            d_mle,
        "d_pca":            d_pca,
    }

    logger.info(
        "  Consensus d_int=%.1f  (TwoNN=%.1f, MLE=%.1f, PCA=%s, agree=%s)",
        consensus or -1,
        d_twonn or -1,
        d_mle or -1,
        str(d_pca),
        agreement,
    )

    return results
