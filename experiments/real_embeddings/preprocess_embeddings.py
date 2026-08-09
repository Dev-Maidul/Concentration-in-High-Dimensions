"""
preprocess_embeddings.py
--------------------------
One-time preprocessing script that generates all processed .npy files
from the raw downloaded datasets.

Run this BEFORE run_embedding_geometry.py.
It loads each dataset, computes intrinsic dimensions, and saves
everything to data/processed/ for fast reuse.

Usage
-----
    cd /mnt/data2/naeem/Geometry-and-Concentration-in-High-Dimensions-main
    python experiments/real_embeddings/preprocess_embeddings.py

    # Skip datasets already processed:
    python experiments/real_embeddings/preprocess_embeddings.py --skip-cached

    # Only process specific datasets:
    python experiments/real_embeddings/preprocess_embeddings.py --datasets glove_50d glove_300d bert_768d
"""

import sys
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/mnt/data2/naeem/Geometry-and-Concentration-in-High-Dimensions-main")
sys.path.insert(0, str(PROJECT_ROOT))

from core.embedding_loader import DATASET_REGISTRY, load_dataset, PROC_DIR
from core.intrinsic_dim import estimate_intrinsic_dim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

INTRINSIC_DIM_CACHE = PROC_DIR / "intrinsic_dimensions.json"


def load_existing_id_cache() -> dict:
    if INTRINSIC_DIM_CACHE.exists():
        with open(INTRINSIC_DIM_CACHE) as fh:
            return json.load(fh)
    return {}


def save_id_cache(cache: dict):
    with open(INTRINSIC_DIM_CACHE, "w") as fh:
        json.dump(cache, fh, indent=2, default=lambda x: float(x)
                  if isinstance(x, (float, int)) else x)
    logger.info("Intrinsic dim cache saved: %s", INTRINSIC_DIM_CACHE)


def preprocess_one(name: str, seed: int = 42,
                    skip_cached: bool = False,
                    id_cache: dict = None) -> dict:
    """
    Load dataset, compute intrinsic dim, return summary.
    """
    if id_cache is None:
        id_cache = {}

    logger.info("-" * 55)
    logger.info("Dataset: %s", name)

    t0 = time.perf_counter()

    # Load (uses cache if available)
    try:
        X = load_dataset(name, seed=seed)
    except FileNotFoundError as e:
        logger.error("  SKIPPED — file not found: %s", e)
        return {"name": name, "status": "file_not_found", "error": str(e)}
    except Exception as e:
        logger.error("  FAILED — %s", e)
        return {"name": name, "status": "error", "error": str(e)}

    elapsed_load = time.perf_counter() - t0
    logger.info("  Loaded: shape=%s  (%.1fs)", X.shape, elapsed_load)

    # Basic statistics
    norms = np.linalg.norm(X, axis=1)
    summary = {
        "name":          name,
        "status":        "ok",
        "shape":         list(X.shape),
        "ambient_dim":   X.shape[1],
        "n_samples":     X.shape[0],
        "norm_mean":     float(np.mean(norms)),
        "norm_std":      float(np.std(norms)),
        "norm_rel_var":  float(np.std(norms) / np.mean(norms)) if np.mean(norms) > 0 else None,
        "data_min":      float(X.min()),
        "data_max":      float(X.max()),
        "has_nan":       bool(np.isnan(X).any()),
        "has_inf":       bool(np.isinf(X).any()),
    }

    # Intrinsic dimension estimation
    if name in id_cache and skip_cached:
        logger.info("  Intrinsic dim: using cached result")
        summary["intrinsic_dim"] = id_cache[name]
    else:
        logger.info("  Estimating intrinsic dimension …")
        t1 = time.perf_counter()

        # Use larger subsample for smaller datasets
        subsample = min(5_000, X.shape[0])

        # Skip correlation dim for very high-dimensional data (slow)
        run_corr = X.shape[1] <= 1000

        id_result = estimate_intrinsic_dim(
            X, subsample=subsample, seed=seed, run_corr_dim=run_corr
        )
        elapsed_id = time.perf_counter() - t1
        logger.info("  Intrinsic dim done (%.1fs)", elapsed_id)

        # Store clean summary
        consensus = id_result["consensus"]
        summary["intrinsic_dim"] = {
            "d_int":            consensus.get("d_int"),
            "d_twonn":          consensus.get("d_twonn"),
            "d_mle":            consensus.get("d_mle"),
            "d_pca":            consensus.get("d_pca"),
            "twonn_mle_agree":  consensus.get("twonn_mle_agree"),
            "full_result":      {
                "twonn":    id_result["twonn"],
                "mle":      id_result["mle"],
                "pca":      id_result["pca"],
                "corr_dim": id_result["corr_dim"],
            }
        }
        id_cache[name] = summary["intrinsic_dim"]

    total_elapsed = time.perf_counter() - t0
    summary["total_elapsed_s"] = total_elapsed

    d_int = summary["intrinsic_dim"].get("d_int")
    logger.info(
        "  Done: d_ambient=%d  d_int=%.1f  agree=%s  (total %.1fs)",
        X.shape[1],
        d_int or -1,
        summary["intrinsic_dim"].get("twonn_mle_agree"),
        total_elapsed,
    )

    return summary


def main():
    parser = argparse.ArgumentParser(description="Preprocess real embedding datasets")
    parser.add_argument(
        "--datasets", nargs="+",
        default=list(DATASET_REGISTRY.keys()),
        help="Datasets to process (default: all)"
    )
    parser.add_argument(
        "--skip-cached", action="store_true",
        help="Skip intrinsic dim computation if already in cache"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PREPROCESSING REAL EMBEDDINGS")
    logger.info("Datasets: %s", args.datasets)
    logger.info("=" * 60)

    id_cache = load_existing_id_cache()
    all_summaries = {}

    for name in args.datasets:
        if name not in DATASET_REGISTRY:
            logger.warning("Unknown dataset '%s', skipping", name)
            continue

        summary = preprocess_one(
            name,
            seed=args.seed,
            skip_cached=args.skip_cached,
            id_cache=id_cache,
        )
        all_summaries[name] = summary

        # Save cache after each dataset (in case of crash)
        save_id_cache(id_cache)

    # Save full preprocessing summary
    summary_path = PROC_DIR / "preprocessing_summary.json"
    with open(summary_path, "w") as fh:
        def _default(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, bool):
                return bool(obj)
            return str(obj)
        json.dump(all_summaries, fh, indent=2, default=_default)
    logger.info("Summary saved: %s", summary_path)

    # Print final summary table
    print("\n" + "=" * 80)
    print(f"{'Dataset':<25} {'Status':>8} {'d_amb':>7} {'d_int':>7} "
          f"{'d_twonn':>9} {'d_mle':>7} {'agree':>7}")
    print("-" * 80)
    for name, s in all_summaries.items():
        if s["status"] != "ok":
            print(f"{name:<25} {'FAILED':>8}")
            continue
        id_s   = s.get("intrinsic_dim", {})
        d_amb  = s["ambient_dim"]
        d_int  = id_s.get("d_int")
        d_2nn  = id_s.get("d_twonn")
        d_mle  = id_s.get("d_mle")
        agree  = id_s.get("twonn_mle_agree")

        def _fmt(v):
            if v is None:
                return "   N/A"
            return f"{v:>7.1f}"

        print(f"{name:<25} {'ok':>8} {d_amb:>7} {_fmt(d_int)} "
              f"{_fmt(d_2nn)} {_fmt(d_mle)} {'yes' if agree else 'no':>7}")
    print("=" * 80)
    print(f"\nProcessed {len(all_summaries)} datasets.")
    print(f"Processed files in: {PROC_DIR}")
    print(f"Intrinsic dim cache: {INTRINSIC_DIM_CACHE}")


if __name__ == "__main__":
    main()
