"""
run_embedding_geometry.py
--------------------------
Section 6: Real Embedding Validation.

Runs the full geometric analysis pipeline on all real datasets,
testing whether the synthetic scaling laws hold when ambient
dimension is replaced by intrinsic dimension d_int.

Key tests per dataset:
  - Does norm variance follow 0.728 * d_int^{-0.505}?
  - Does distance contrast follow 17.35 * d_int^{-0.623}?
  - Does the hubness Gini threshold (0.75) predict observed Gini at d_int?
  - Does NN ratio match predictions?
  - For BERT: does training geometry counteract expected pathologies?
  - For scRNA: does heavy-tail amplification match Law 6?

Saves:
  results/raw/real_embeddings/embedding_geometry_full.json
  results/raw/real_embeddings/scaling_law_validation.json

Usage
-----
    python experiments/real_embeddings/run_embedding_geometry.py
    python experiments/real_embeddings/run_embedding_geometry.py --fast
    python experiments/real_embeddings/run_embedding_geometry.py \
        --datasets glove_50d glove_100d glove_200d glove_300d bert_768d
"""

import sys
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import skew

PROJECT_ROOT = Path("/mnt/data2/naeem/Geometry-and-Concentration-in-High-Dimensions-main")
sys.path.insert(0, str(PROJECT_ROOT))

from core.embedding_loader import DATASET_REGISTRY, load_dataset, PROC_DIR
from core.intrinsic_dim import estimate_intrinsic_dim
from core.metrics import (
    compute_norm_statistics,
    compute_distance_statistics,
    compute_hubness_statistics,
)
from core.projection_methods import (
    gaussian_projection,
    sparse_projection,
    structured_projection,
)
from core.scaling_analysis import fit_power_law

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results" / "raw" / "real_embeddings"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Synthetic scaling law constants (from paper) ──────────────────────────────
LAW_NORM_A    = 0.728
LAW_NORM_B    = -0.505
LAW_DIST_A    = 17.35
LAW_DIST_B    = -0.623
LAW_HUBNESS_THRESHOLD_D = 100   # d_int where Gini > 0.75
LAW_NN_RATIO_AT_100     = 0.77  # r_NN at d=100


def predict_from_law(d_int: float) -> Dict:
    """
    Given intrinsic dimension d_int, predict geometric quantities
    from the empirical scaling laws.
    """
    if d_int is None or d_int <= 0:
        return {}
    return {
        "pred_norm_rel_var":  LAW_NORM_A * (d_int ** LAW_NORM_B),
        "pred_dist_contrast": LAW_DIST_A * (d_int ** LAW_DIST_B),
        "pred_hubness_severe": d_int >= LAW_HUBNESS_THRESHOLD_D,
        "pred_nn_ratio":      float(1 - 0.4 * (d_int ** -0.35)),  # empirical approximation
    }


def _gini(counts: np.ndarray) -> float:
    n = len(counts)
    if n == 0 or counts.sum() == 0:
        return 0.0
    sorted_c = np.sort(counts)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * sorted_c).sum() - (n + 1) * sorted_c.sum()) /
                 (n * sorted_c.sum()))


def analyse_single_embedding(
    X: np.ndarray,
    d_int: Optional[float],
    name: str,
    n_main: int = 10_000,
    n_dist: int = 1_000,
    k_nn: int = 10,
    seed: int = 42,
) -> Dict:
    """
    Full geometric analysis of one embedding matrix.

    Parameters
    ----------
    X     : (N, d_ambient) float32
    d_int : estimated intrinsic dimension (can be None)
    name  : dataset name for logging
    n_main: subsample size for norm/hubness analysis
    n_dist: subsample size for pairwise distance analysis
    k_nn  : k for hubness kNN graph
    seed  : reproducibility
    """
    rng = np.random.default_rng(seed)
    N, d_amb = X.shape

    # Subsample
    idx_main = rng.choice(N, size=min(n_main, N), replace=False)
    X_main   = X[idx_main].astype(np.float64)

    idx_dist = rng.choice(len(X_main), size=min(n_dist, len(X_main)), replace=False)
    X_dist   = X_main[idx_dist]

    results = {
        "name":       name,
        "d_ambient":  d_amb,
        "d_int":      d_int,
        "n_used":     len(X_main),
        "n_dist":     len(X_dist),
    }

    # ── 1. Norm concentration ──────────────────────────────────────────────
    logger.info("  [norm] n=%d d=%d", len(X_main), d_amb)
    norms    = np.linalg.norm(X_main, axis=1)
    mean_n   = float(np.mean(norms))
    std_n    = float(np.std(norms, ddof=1))
    rel_var  = float(std_n / mean_n) if mean_n > 0 else np.nan
    shell_t  = float((norms.max() - norms.min()) / mean_n) if mean_n > 0 else np.nan

    results["norm"] = {
        "mean_norm":        mean_n,
        "std_norm":         std_n,
        "relative_variance": rel_var,
        "shell_thickness":  shell_t,
    }

    # ── 2. Distance geometry ───────────────────────────────────────────────
    logger.info("  [distance] n_dist=%d", len(X_dist))
    n_d  = len(X_dist)
    diff = X_dist[:, None, :] - X_dist[None, :, :]
    dmat = np.sqrt((diff ** 2).sum(axis=2))

    upper_idx = np.triu_indices(n_d, k=1)
    upper     = dmat[upper_idx]

    mean_d    = float(np.mean(upper))
    std_d     = float(np.std(upper, ddof=1))
    min_d     = float(upper.min())
    max_d     = float(upper.max())
    rel_cont  = float((max_d - min_d) / min_d) if min_d > 0 else np.nan

    # NN ratio
    np.fill_diagonal(dmat, np.inf)
    nn_dists   = dmat.min(axis=1)
    mean_dists = np.where(
        np.isfinite(dmat).any(axis=1),
        np.where(np.isfinite(dmat), dmat, 0).sum(axis=1) / (n_d - 1),
        np.nan
    )
    valid_nn = (nn_dists > 0) & (mean_dists > 0) & np.isfinite(nn_dists) & np.isfinite(mean_dists)
    nn_ratio = float(np.mean(nn_dists[valid_nn] / mean_dists[valid_nn])) if valid_nn.any() else np.nan

    results["distance"] = {
        "mean_dist":    mean_d,
        "std_dist":     std_d,
        "min_dist":     min_d,
        "max_dist":     max_d,
        "rel_contrast": rel_cont,
        "nn_ratio":     nn_ratio,
    }

    # ── 3. Hubness ─────────────────────────────────────────────────────────
    logger.info("  [hubness] k=%d n=%d", k_nn, len(X_dist))
    np.fill_diagonal(dmat, np.inf)
    nn_counts = np.zeros(n_d, dtype=int)
    for i in range(n_d):
        k_idx = np.argpartition(dmat[i], k_nn)[:k_nn]
        nn_counts[k_idx] += 1

    gini       = _gini(nn_counts)
    hub_skew   = float(skew(nn_counts))
    max_hub    = int(nn_counts.max())
    mean_hub   = float(nn_counts.mean())

    results["hubness"] = {
        "gini":       gini,
        "skewness":   hub_skew,
        "max_count":  max_hub,
        "mean_count": mean_hub,
        "nn_counts":  nn_counts.tolist(),
    }

    # ── 4. JL Projections ─────────────────────────────────────────────────
    logger.info("  [projections] d=%d", d_amb)
    n_proj   = min(500, len(X_dist))
    X_proj   = X_dist[:n_proj]
    orig_dm  = np.sqrt(((X_proj[:, None] - X_proj[None]) ** 2).sum(2))
    orig_pairs = orig_dm[np.triu_indices(n_proj, k=1)]

    proj_results = {}
    for ratio in [0.10, 0.25, 0.50]:
        k = max(1, int(round(ratio * d_amb)))
        if k >= d_amb:
            continue
        k_key = f"ratio{ratio:.2f}"
        proj_results[k_key] = {}

        for method_name, proj_fn in [
            ("gaussian",   gaussian_projection),
            ("sparse",     sparse_projection),
            ("structured", structured_projection),
        ]:
            try:
                proj_out = proj_fn(X_proj, k,
                                    seed=int(rng.integers(1_000_000)))
                # Handle both (X_proj, R) tuple and plain array returns
                if isinstance(proj_out, tuple):
                    X_red = proj_out[0]
                else:
                    X_red = proj_out

                proj_dm    = np.sqrt(((X_red[:, None] - X_red[None]) ** 2).sum(2))
                proj_pairs = proj_dm[np.triu_indices(n_proj, k=1)]

                mask = orig_pairs > 1e-10
                distortions = proj_pairs[mask] / orig_pairs[mask]

                proj_results[k_key][method_name] = {
                    "mean_distortion": float(np.mean(distortions)),
                    "std_distortion":  float(np.std(distortions)),
                    "fail_0.1":        float(np.mean(np.abs(distortions - 1) > 0.1)),
                    "fail_0.2":        float(np.mean(np.abs(distortions - 1) > 0.2)),
                }
            except Exception as e:
                logger.warning("  Projection failed (%s, k=%d): %s", method_name, k, e)
                proj_results[k_key][method_name] = None

    results["projections"] = proj_results

    # ── 5. Compare to scaling law predictions ─────────────────────────────
    predictions = predict_from_law(d_int)
    observed = {
        "norm_rel_var":  rel_var,
        "dist_contrast": rel_cont,
        "hubness_gini":  gini,
        "nn_ratio":      nn_ratio,
    }

    deviations = {}
    if d_int is not None:
        for key, pred_key in [
            ("norm_rel_var",  "pred_norm_rel_var"),
            ("dist_contrast", "pred_dist_contrast"),
        ]:
            obs = observed[key]
            pred = predictions.get(pred_key)
            if obs is not None and pred is not None and pred > 0:
                deviations[key] = float(100 * (obs - pred) / pred)

    results["law_comparison"] = {
        "d_int_used":  d_int,
        "predictions": predictions,
        "observed":    {k: (float(v) if v is not None else None)
                        for k, v in observed.items()},
        "deviations_pct": deviations,
    }

    return results


def run_with_multiple_trials(
    name: str,
    X: np.ndarray,
    d_int: Optional[float],
    n_trials: int = 5,
    seed: int = 42,
) -> Dict:
    """
    Run geometric analysis with multiple random subsampling trials
    to get uncertainty estimates.
    """
    trial_results = []
    for t in range(n_trials):
        t_seed = seed + t * 1000
        logger.info("  Trial %d/%d (seed=%d) …", t + 1, n_trials, t_seed)
        t0 = time.perf_counter()
        res = analyse_single_embedding(X, d_int, name, seed=t_seed)
        logger.info("    Done in %.1fs", time.perf_counter() - t0)
        trial_results.append(res)

    # Aggregate scalar leaves across trials
    def _agg(items):
        if isinstance(items[0], dict):
            return {k: _agg([x[k] for x in items if k in x])
                    for k in items[0]}
        elif isinstance(items[0], (int, float)) and not isinstance(items[0], bool):
            vals = [x for x in items if x is not None and np.isfinite(x)]
            if not vals:
                return None
            return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        elif isinstance(items[0], list):
            return items[0]  # keep first trial's list
        else:
            return items[0]

    aggregated = _agg(trial_results)
    aggregated["n_trials"] = n_trials
    return aggregated


def run_all(dataset_names: List[str] = None,
             n_trials: int = 5,
             seed: int = 42) -> Dict:
    """
    Run the full Section 6 experiment on all specified datasets.
    """
    if dataset_names is None:
        dataset_names = list(DATASET_REGISTRY.keys())

    # Load intrinsic dimension cache
    id_cache_path = PROC_DIR / "intrinsic_dimensions.json"
    if id_cache_path.exists():
        with open(id_cache_path) as fh:
            id_cache = json.load(fh)
        logger.info("Loaded intrinsic dim cache: %d entries", len(id_cache))
    else:
        id_cache = {}
        logger.warning("No intrinsic dim cache found — will estimate on the fly")

    master = {}

    for name in dataset_names:
        if name not in DATASET_REGISTRY:
            logger.warning("Unknown dataset '%s', skipping", name)
            continue

        logger.info("=" * 60)
        logger.info("DATASET: %s", name)
        logger.info("=" * 60)

        try:
            X = load_dataset(name, seed=seed)
        except FileNotFoundError as e:
            logger.error("  Skipping — %s", e)
            master[name] = {"status": "file_not_found", "error": str(e)}
            continue

        # Get intrinsic dimension
        if name in id_cache:
            d_int = id_cache[name].get("d_int")
            logger.info("  d_int from cache: %.1f", d_int or -1)
        else:
            logger.info("  Estimating d_int …")
            id_res = estimate_intrinsic_dim(
                X, subsample=min(5_000, X.shape[0]), seed=seed,
                run_corr_dim=X.shape[1] <= 1000
            )
            d_int = id_res["consensus"]["d_int"]
            id_cache[name] = id_res["consensus"]
            # Save updated cache
            with open(id_cache_path, "w") as fh:
                json.dump(id_cache, fh, indent=2,
                          default=lambda x: float(x)
                          if isinstance(x, (float, int)) else x)

        result = run_with_multiple_trials(
            name=name,
            X=X,
            d_int=d_int,
            n_trials=n_trials,
            seed=seed,
        )
        result["status"] = "ok"
        master[name] = result

        # Save per-dataset result
        out = RESULTS_DIR / f"{name}_geometry.json"
        with open(out, "w") as fh:
            def _jdefault(obj):
                if isinstance(obj, (np.floating, np.integer)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return str(obj)
            json.dump(result, fh, indent=2, default=_jdefault)
        logger.info("  Saved: %s", out)

    # Combined file
    combined_out = RESULTS_DIR / "embedding_geometry_full.json"
    with open(combined_out, "w") as fh:
        def _jd2(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)
        json.dump(master, fh, indent=2, default=_jd2)
    logger.info("Combined results: %s", combined_out)

    # Build scaling law validation summary
    build_scaling_validation(master)

    return master


def build_scaling_validation(master: Dict):
    """
    Build a clean validation summary comparing predictions to observations
    for every dataset. Saved to scaling_law_validation.json.
    """
    def _get(d, *keys):
        cur = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        if isinstance(cur, dict) and "mean" in cur:
            return cur["mean"]
        return cur

    validation = {}
    for name, result in master.items():
        if result.get("status") != "ok":
            continue

        d_int  = _get(result, "d_int")
        d_amb  = _get(result, "d_ambient")
        preds  = predict_from_law(d_int)

        obs_norm = _get(result, "norm", "relative_variance")
        obs_dist = _get(result, "distance", "rel_contrast")
        obs_gini = _get(result, "hubness", "gini")
        obs_nn   = _get(result, "distance", "nn_ratio")

        entry = {
            "d_ambient":    d_amb,
            "d_int":        d_int,
            "observed":     {
                "norm_rel_var":  obs_norm,
                "dist_contrast": obs_dist,
                "hubness_gini":  obs_gini,
                "nn_ratio":      obs_nn,
            },
            "predicted": preds,
        }

        # Compute deviations
        for obs_key, pred_key in [
            ("norm_rel_var",  "pred_norm_rel_var"),
            ("dist_contrast", "pred_dist_contrast"),
        ]:
            obs  = entry["observed"][obs_key]
            pred = preds.get(pred_key)
            if obs is not None and pred is not None and pred > 0:
                entry[f"{obs_key}_dev_pct"] = float(100 * (obs - pred) / pred)

        validation[name] = entry

    val_out = RESULTS_DIR / "scaling_law_validation.json"
    with open(val_out, "w") as fh:
        json.dump(validation, fh, indent=2,
                  default=lambda x: float(x)
                  if isinstance(x, (float, int)) else str(x))
    logger.info("Scaling law validation: %s", val_out)

    # Print summary
    print("\n" + "=" * 100)
    print(f"{'Dataset':<25} {'d_amb':>6} {'d_int':>7} "
          f"{'obs σ_rel':>10} {'pred σ_rel':>10} {'dev%':>7} "
          f"{'obs Crel':>10} {'pred Crel':>10} {'dev%':>7} "
          f"{'Gini':>7}")
    print("-" * 100)
    for name, v in validation.items():
        d_amb  = v["d_ambient"] or -1
        d_int  = v["d_int"]    or -1
        o      = v["observed"]
        p      = v["predicted"]

        def _f(x):
            return f"{x:.4f}" if x is not None else "  N/A "

        dev_n = v.get("norm_rel_var_dev_pct")
        dev_d = v.get("dist_contrast_dev_pct")

        print(
            f"{name:<25} {d_amb:>6.0f} {d_int:>7.1f} "
            f"{_f(o['norm_rel_var']):>10} {_f(p.get('pred_norm_rel_var')):>10} "
            f"{f'{dev_n:+.1f}%' if dev_n is not None else '   N/A':>7} "
            f"{_f(o['dist_contrast']):>10} {_f(p.get('pred_dist_contrast')):>10} "
            f"{f'{dev_d:+.1f}%' if dev_d is not None else '   N/A':>7} "
            f"{_f(o['hubness_gini']):>7}"
        )
    print("=" * 100)

    return validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Section 6: Real embedding geometry")
    parser.add_argument(
        "--datasets", nargs="+",
        default=list(DATASET_REGISTRY.keys()),
        help="Datasets to run (default: all)"
    )
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true",
                        help="Use n_trials=2 for quick testing")
    args = parser.parse_args()

    n_t = 2 if args.fast else args.n_trials
    run_all(args.datasets, n_trials=n_t, seed=args.seed)
