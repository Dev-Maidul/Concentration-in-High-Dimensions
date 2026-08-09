"""
analyze_discrepancies.py
-------------------------
Section 6.4: Analysis of discrepancies between synthetic predictions
and real embedding observations.

Key questions:
  1. For BERT: is hubness lower than predicted? Is NN accuracy higher?
     (Training counteracts geometric pathology)
  2. For scRNA: is hubness worse than predicted?
     (Heavy-tail amplification — connects to Law 6)
  3. For GloVe at multiple dimensions: does d_int correctly rescale laws?
  4. Are deviations systematic or random?

Saves:
  results/raw/real_embeddings/discrepancy_analysis.json
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

PROJECT_ROOT = Path("/mnt/data2/naeem/Geometry-and-Concentration-in-High-Dimensions-main")
sys.path.insert(0, str(PROJECT_ROOT))

from core.embedding_loader import load_dataset, PROC_DIR
from core.intrinsic_dim import pca_effective_dim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results" / "raw" / "real_embeddings"


def _load_geometry(name: str) -> Optional[Dict]:
    """Load per-dataset geometry result."""
    p = RESULTS_DIR / f"{name}_geometry.json"
    if not p.exists():
        logger.warning("Missing geometry file: %s", p)
        return None
    with open(p) as fh:
        return json.load(fh)


def _get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    if isinstance(cur, dict) and "mean" in cur:
        return cur["mean"]
    return cur


def analyse_bert_discrepancy(seed: int = 42) -> Dict:
    """
    BERT analysis: test whether training counteracts geometric pathology.

    Expected finding: BERT hubness is lower than synthetic prediction
    at the same d_int, because contrastive/fine-tuning objectives
    explicitly push dissimilar things apart.

    We test this by:
    1. Computing predicted Gini at d_int(BERT)
    2. Observing actual Gini
    3. Measuring NN semantic accuracy (do NNs share topics/sentiment?)
       as a proxy for meaningful geometry.
    """
    logger.info("BERT discrepancy analysis …")
    geo = _load_geometry("bert_768d")
    if geo is None:
        return {"status": "no_data"}

    d_int = _get(geo, "d_int")
    obs_gini = _get(geo, "hubness", "gini")
    obs_nn_ratio = _get(geo, "distance", "nn_ratio")
    obs_contrast = _get(geo, "distance", "rel_contrast")

    # Synthetic predictions at d_int
    from experiments.real_embeddings.run_embedding_geometry import predict_from_law
    preds = predict_from_law(d_int)

    # Gini deviation: negative means BERT hubness is LOWER than predicted
    gini_dev = None
    if obs_gini is not None and preds.get("pred_hubness_severe") is not None:
        # Compare to synthetic Gaussian at same d_int
        # From Law 3: at d~100, Gini ~ 0.75; at d~200 Gini ~ 0.81
        # Linearly interpolate
        if d_int is not None:
            synth_gini_pred = 0.131 + (0.849 - 0.131) * min(1.0, d_int / 2000)
            gini_dev = float(100 * (obs_gini - synth_gini_pred) / synth_gini_pred)

    result = {
        "dataset":          "bert_768d",
        "d_int":            d_int,
        "d_ambient":        768,
        "obs_gini":         obs_gini,
        "obs_nn_ratio":     obs_nn_ratio,
        "obs_contrast":     obs_contrast,
        "predictions":      preds,
        "gini_deviation_pct": gini_dev,
        "interpretation": (
            "If gini_deviation_pct < 0, BERT training has reduced hubness "
            "below what synthetic geometry predicts at the same intrinsic "
            "dimension. This suggests training explicitly shaped geometry "
            "to counteract distance pathology."
        )
    }

    logger.info(
        "BERT: d_int=%.1f  obs_gini=%.3f  gini_dev=%.1f%%",
        d_int or -1, obs_gini or -1, gini_dev or 0
    )
    return result


def analyse_glove_multidim(seed: int = 42) -> Dict:
    """
    GloVe multi-dimension analysis: test whether scaling laws hold
    as ambient dimension increases with fixed semantic content.

    GloVe has the same vocabulary in 50d, 100d, 200d, 300d versions.
    The intrinsic dimension of the semantic space is fixed, but ambient
    dimension grows. This gives us a controlled test of:
      - Does Crel(d) follow the -0.623 law as d_amb increases?
      - Does intrinsic dimension shift when training provides more capacity?
    """
    logger.info("GloVe multi-dimension analysis …")

    results = {}
    for dim in [50, 100, 200, 300]:
        name = f"glove_{dim}d"
        geo  = _load_geometry(name)
        if geo is None:
            continue

        results[dim] = {
            "d_ambient":    dim,
            "d_int":        _get(geo, "d_int"),
            "rel_contrast": _get(geo, "distance", "rel_contrast"),
            "norm_rel_var": _get(geo, "norm", "relative_variance"),
            "hubness_gini": _get(geo, "hubness", "gini"),
            "nn_ratio":     _get(geo, "distance", "nn_ratio"),
        }

    # Fit power law to contrast vs ambient dimension
    dims   = np.array([d for d in [50, 100, 200, 300] if d in results])
    contrasts = np.array([results[d]["rel_contrast"] for d in dims
                           if results[d]["rel_contrast"] is not None])

    fitted_exp = None
    if len(dims) >= 3 and len(contrasts) == len(dims):
        from scipy import stats
        valid = (dims > 0) & (contrasts > 0) & np.isfinite(contrasts)
        if valid.sum() >= 3:
            slope, _, r, _, se = stats.linregress(
                np.log(dims[valid]), np.log(contrasts[valid])
            )
            fitted_exp = float(slope)

    # Fit power law to contrast vs intrinsic dimension
    d_int_vals = np.array([results[d]["d_int"] for d in dims
                            if results[d]["d_int"] is not None and
                            d in results and results[d]["rel_contrast"] is not None])
    contrasts_for_int = np.array([results[d]["rel_contrast"] for d in dims
                                    if results[d]["d_int"] is not None and
                                    d in results and results[d]["rel_contrast"] is not None])

    fitted_exp_int = None
    if len(d_int_vals) >= 3:
        from scipy import stats
        valid = (d_int_vals > 0) & (contrasts_for_int > 0) & np.isfinite(contrasts_for_int)
        if valid.sum() >= 3:
            slope, _, _, _, _ = stats.linregress(
                np.log(d_int_vals[valid]), np.log(contrasts_for_int[valid])
            )
            fitted_exp_int = float(slope)

    summary = {
        "per_dimension":         results,
        "fitted_exponent_vs_d_amb": fitted_exp,
        "fitted_exponent_vs_d_int": fitted_exp_int,
        "target_exponent":         -0.623,
        "interpretation": (
            f"Exponent vs ambient d = {fitted_exp:.3f if fitted_exp else 'N/A'}. "
            f"Exponent vs intrinsic d = {fitted_exp_int:.3f if fitted_exp_int else 'N/A'}. "
            "If exponent vs d_int is closer to -0.623 than vs d_amb, "
            "intrinsic dimension is the right scale for the law."
        )
    }
    logger.info("GloVe: exp_vs_d_amb=%.3f  exp_vs_d_int=%.3f",
                fitted_exp or -999, fitted_exp_int or -999)
    return summary


def analyse_scrna_heavy_tail(seed: int = 42) -> Dict:
    """
    scRNA analysis: test heavy-tail amplification (Law 6).

    scRNA data is sparse and heavy-tailed (many near-zero counts,
    occasional very high expression). We predict:
    - Hubness is WORSE than Gaussian prediction at same d_int
    - Deviation magnitude is consistent with Laplace-like tail amplification
      (12-15% worse per our Law 6)
    """
    logger.info("scRNA heavy-tail analysis …")
    geo = _load_geometry("scrna_pbmc")
    if geo is None:
        return {"status": "no_data"}

    d_int    = _get(geo, "d_int")
    obs_gini = _get(geo, "hubness", "gini")
    obs_skew = _get(geo, "hubness", "skewness")
    obs_max  = _get(geo, "hubness", "max_count")

    # Load raw data for tail analysis
    try:
        X = load_dataset("scrna_pbmc", seed=seed)
        # Measure kurtosis of marginal distributions
        from scipy.stats import kurtosis
        kurt_vals = [kurtosis(X[:, j], fisher=True) for j in
                     range(min(200, X.shape[1]))]
        mean_kurt = float(np.mean(kurt_vals))
        is_heavy_tail = mean_kurt > 1.0  # excess kurtosis > 1 = heavy tail
    except Exception:
        mean_kurt = None
        is_heavy_tail = None

    # Gaussian prediction at same d_int
    synth_gini_gaussian = None
    if d_int is not None:
        synth_gini_gaussian = 0.131 + (0.849 - 0.131) * min(1.0, d_int / 2000)

    # Laplace prediction (12-15% worse per Law 6)
    synth_gini_laplace = None
    if synth_gini_gaussian is not None:
        synth_gini_laplace = synth_gini_gaussian * 1.135  # 13.5% amplification

    gini_dev_from_gaussian = None
    gini_dev_from_laplace  = None
    if obs_gini is not None and synth_gini_gaussian is not None:
        gini_dev_from_gaussian = float(100 * (obs_gini - synth_gini_gaussian)
                                        / synth_gini_gaussian)
    if obs_gini is not None and synth_gini_laplace is not None:
        gini_dev_from_laplace = float(100 * (obs_gini - synth_gini_laplace)
                                       / synth_gini_laplace)

    result = {
        "dataset":                    "scrna_pbmc",
        "d_int":                      d_int,
        "obs_gini":                   obs_gini,
        "obs_skewness":               obs_skew,
        "obs_max_hub_count":          obs_max,
        "mean_marginal_kurtosis":     mean_kurt,
        "is_heavy_tail":              is_heavy_tail,
        "synth_gini_gaussian_pred":   synth_gini_gaussian,
        "synth_gini_laplace_pred":    synth_gini_laplace,
        "gini_dev_from_gaussian_pct": gini_dev_from_gaussian,
        "gini_dev_from_laplace_pct":  gini_dev_from_laplace,
        "interpretation": (
            "Positive gini_dev_from_gaussian_pct means scRNA hubness "
            "exceeds Gaussian prediction — consistent with Law 6 "
            "(heavy-tail amplification). Deviation close to 12-15% "
            "would match Laplace-level amplification."
        )
    }
    logger.info(
        "scRNA: d_int=%.1f  obs_gini=%.3f  dev_from_gaussian=%.1f%%",
        d_int or -1, obs_gini or -1, gini_dev_from_gaussian or 0
    )
    return result


def run_all_discrepancy_analyses(seed: int = 42) -> Dict:
    """Run all discrepancy analyses and save results."""
    logger.info("=" * 60)
    logger.info("SECTION 6.4: Discrepancy analysis")
    logger.info("=" * 60)

    results = {
        "bert":        analyse_bert_discrepancy(seed=seed),
        "glove_multi": analyse_glove_multidim(seed=seed),
        "scrna":       analyse_scrna_heavy_tail(seed=seed),
    }

    # Print interpretation summary
    print("\n" + "=" * 70)
    print("DISCREPANCY ANALYSIS SUMMARY")
    print("=" * 70)

    bert = results["bert"]
    if bert.get("obs_gini") is not None:
        dev = bert.get("gini_deviation_pct")
        direction = "LOWER" if (dev is not None and dev < 0) else "HIGHER"
        print(f"\nBERT (d_int={bert.get('d_int', '?'):.1f}):")
        print(f"  Observed Gini = {bert['obs_gini']:.3f}")
        print(f"  Deviation from synthetic = {dev:+.1f}% ({direction} than predicted)")
        if dev is not None and dev < -10:
            print("  → Training has significantly counteracted geometric pathology")
        elif dev is not None and abs(dev) <= 10:
            print("  → Training has modest effect on geometry")

    scrna = results["scrna"]
    if scrna.get("obs_gini") is not None:
        dev_g = scrna.get("gini_dev_from_gaussian_pct")
        dev_l = scrna.get("gini_dev_from_laplace_pct")
        print(f"\nscRNA PBMC (d_int={scrna.get('d_int', '?')}):")
        print(f"  Observed Gini = {scrna.get('obs_gini', '?'):.3f}")
        print(f"  Deviation from Gaussian prediction = {dev_g:+.1f}%"
              if dev_g is not None else "  Deviation: N/A")
        print(f"  Deviation from Laplace prediction  = {dev_l:+.1f}%"
              if dev_l is not None else "  Deviation: N/A")
        if dev_g is not None and dev_g > 10:
            print("  → Consistent with heavy-tail amplification (Law 6)")

    glove = results["glove_multi"]
    exp_amb = glove.get("fitted_exponent_vs_d_amb")
    exp_int = glove.get("fitted_exponent_vs_d_int")
    print(f"\nGloVe multi-dim:")
    print(f"  Exponent vs d_ambient   = {exp_amb:.3f}" if exp_amb else "  N/A")
    print(f"  Exponent vs d_int       = {exp_int:.3f}" if exp_int else "  N/A")
    print(f"  Target (synthetic)      = -0.623")
    if exp_int and abs(exp_int - (-0.623)) < abs((exp_amb or 0) - (-0.623)):
        print("  → d_int rescaling improves law accuracy ✓")

    print("=" * 70)

    # Save
    out = RESULTS_DIR / "discrepancy_analysis.json"
    with open(out, "w") as fh:
        def _jd(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)
        json.dump(results, fh, indent=2, default=_jd)
    logger.info("Saved: %s", out)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_all_discrepancy_analyses(seed=args.seed)
