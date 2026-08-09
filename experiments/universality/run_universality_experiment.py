"""
run_universality_experiment.py
--------------------------------
Section 5 experiments: Universality of the -0.623 exponent and
heuristic derivation supporting it.

Runs:
  5.1 — Universality across 10 distributions with bootstrap CIs
  5.2 — Empirical verification of Corr(Dij^2, Dik^2) = 1/4
  5.2 — Exponent vs sample size n (robustness check for heuristic)
  
Saves results to:
  results/raw/universality_results.pkl
  results/raw/dependence_results.pkl
  results/raw/exponent_vs_n_results.pkl
"""

import sys
import os
import logging
import pickle
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config.global_config import CONFIG, set_seed
from core.reproducibility import setup_logging, save_results
from core.dependence_analysis import (
    analytical_correlation,
    run_correlation_experiment,
    effective_independent_distances,
    exponent_vs_n,
    universality_test,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results" / "raw"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_section5_1(seed: int = 42, n_trials: int = 50) -> dict:
    """
    Section 5.1: Universality of -0.623 across 10 distributions.

    Extends the original 4 distributions to 10, fits power-law exponents
    with bootstrap confidence intervals, and performs t-test against
    null hypothesis of -0.5.
    """
    logger.info("=" * 60)
    logger.info("SECTION 5.1: Universality experiment")
    logger.info("=" * 60)

    results = universality_test(
        distributions=[
            "gaussian", "uniform", "laplace", "student_t3",
            "student_t5", "student_t10", "exponential", "beta22",
            "mixed_gaussian", "cauchy_truncated"
        ],
        dimensions=[50, 100, 200, 500, 1000, 2000],
        n_samples=10_000,
        n_trials=n_trials,
        seed=seed,
    )

    out = RESULTS_DIR / "universality_results.pkl"
    with open(out, "wb") as fh:
        pickle.dump(results, fh)
    logger.info("Saved: %s", out)

    # Print summary table
    print("\n" + "=" * 75)
    print(f"{'Distribution':<20} {'Exponent':>10} {'Std Err':>9} "
          f"{'CI low':>9} {'CI high':>9} {'R²':>7}")
    print("-" * 75)
    for dist, d_res in results["data"].items():
        exp  = d_res["fitted_exponent"]
        se   = d_res["std_error"]
        cil  = d_res["ci_95_low"]
        cih  = d_res["ci_95_high"]
        r2   = d_res["r_squared"]
        if exp is not None:
            print(f"{dist:<20} {exp:>10.4f} {se:>9.4f} "
                  f"{cil:>9.4f} {cih:>9.4f} {r2:>7.4f}")

    s = results["summary"]
    print("-" * 75)
    print(f"{'MEAN':<20} {s['mean_exponent']:>10.4f} {s['std_exponent']:>9.4f}")
    print(f"\nTarget: -0.623   |  t-test vs -0.5: p = {s['t_test_vs_minus05']:.2e}")
    print("=" * 75)

    return results


def run_section5_2_correlation(seed: int = 42, n_trials: int = 100) -> dict:
    """
    Section 5.2: Empirical verification of Corr(Dij^2, Dik^2) = 1/4.

    Verifies the key analytical result across multiple dimensions to
    show the correlation is dimension-independent (as predicted).
    """
    logger.info("=" * 60)
    logger.info("SECTION 5.2a: Correlation verification")
    logger.info("=" * 60)

    # Analytical result
    analytical = analytical_correlation()
    logger.info("Analytical Corr(Dij^2, Dik^2) = %.2f",
                analytical["analytical_correlation_D2"])

    # Empirical verification
    results = run_correlation_experiment(
        dimensions=[5, 10, 20, 50, 100, 200, 500, 1000],
        n_trials=n_trials,
        seed=seed,
    )

    out = RESULTS_DIR / "dependence_results.pkl"
    with open(out, "wb") as fh:
        pickle.dump(results, fh)
    logger.info("Saved: %s", out)

    # Print summary
    print("\n" + "=" * 55)
    print(f"{'Dim':>6} {'Empirical Corr':>15} {'Analytical':>12} {'Diff':>8}")
    print("-" * 55)
    analytical_val = results["analytical"]["analytical_correlation_D2"]
    for d, res in results["empirical"].items():
        diff = res["mean_corr"] - analytical_val
        print(f"{d:>6} {res['mean_corr']:>12.4f} ± {res['std_corr']:.4f} "
              f"{analytical_val:>12.4f} {diff:>8.4f}")
    print("=" * 55)

    return results


def run_section5_2_effective_n(seed: int = 42, n_trials: int = 30) -> dict:
    """
    Section 5.2: Exponent vs sample size test.

    Tests whether the exponent varies with n in the direction predicted
    by the effective-independence model. This is the key empirical check
    for whether our heuristic account is correct.
    """
    logger.info("=" * 60)
    logger.info("SECTION 5.2b: Exponent vs sample size")
    logger.info("=" * 60)

    # Theoretical predictions for each n
    n_values = [500, 1000, 5000, 10_000, 50_000]
    print("\nTheoretical predictions from effective-independence model:")
    print(f"{'n':>8} {'n_total':>12} {'n_eff':>12} {'predicted_exp':>15}")
    print("-" * 55)
    for n in n_values:
        pred = effective_independent_distances(n, rho=0.25)
        print(f"{n:>8} {pred['n_total']:>12.0f} {pred['n_eff']:>12.0f} "
              f"{pred['predicted_exponent']:>15.4f}")

    # Empirical test
    results = exponent_vs_n(
        n_values=n_values,
        dimensions=[50, 100, 200, 500, 1000],
        n_trials=n_trials,
        seed=seed,
    )

    out = RESULTS_DIR / "exponent_vs_n_results.pkl"
    with open(out, "wb") as fh:
        pickle.dump(results, fh)
    logger.info("Saved: %s", out)

    # Print comparison
    print("\n" + "=" * 60)
    print(f"{'n':>8} {'Empirical exp':>15} {'Predicted exp':>15} {'Gap':>8}")
    print("-" * 60)
    for n in n_values:
        d_res   = results["data"][n]
        emp_exp = d_res["fitted_exponent"]
        pred_exp = d_res["predicted_exponent"]
        if emp_exp is not None:
            gap = emp_exp - pred_exp
            print(f"{n:>8} {emp_exp:>15.4f} {pred_exp:>15.4f} {gap:>8.4f}")
    print("=" * 60)

    return results


def main(fast: bool = False):
    """
    Run all Section 5 experiments.

    Parameters
    ----------
    fast : bool
        If True, use fewer trials for quick testing (not for paper)
    """
    set_seed(CONFIG.RANDOM_SEED)

    n_trials_uni  = 10 if fast else 50
    n_trials_corr = 20 if fast else 100
    n_trials_n    = 5  if fast else 30

    logger.info("Starting Section 5 experiments (fast=%s)", fast)

    # 5.1 Universality
    uni_results = run_section5_1(
        seed=CONFIG.RANDOM_SEED,
        n_trials=n_trials_uni
    )

    # 5.2a Correlation verification
    corr_results = run_section5_2_correlation(
        seed=CONFIG.RANDOM_SEED,
        n_trials=n_trials_corr
    )

    # 5.2b Exponent vs n
    n_results = run_section5_2_effective_n(
        seed=CONFIG.RANDOM_SEED,
        n_trials=n_trials_n
    )

    logger.info("All Section 5 experiments complete.")
    return {
        "universality": uni_results,
        "correlation":  corr_results,
        "exponent_vs_n": n_results,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Section 5 universality experiments")
    parser.add_argument("--fast", action="store_true",
                        help="Run with fewer trials for quick testing")
    parser.add_argument("--part", type=str, default="all",
                        choices=["all", "5.1", "5.2a", "5.2b"],
                        help="Which part to run")
    args = parser.parse_args()

    if args.part == "all":
        main(fast=args.fast)
    elif args.part == "5.1":
        run_section5_1(seed=CONFIG.RANDOM_SEED,
                       n_trials=10 if args.fast else 50)
    elif args.part == "5.2a":
        run_section5_2_correlation(seed=CONFIG.RANDOM_SEED,
                                    n_trials=20 if args.fast else 100)
    elif args.part == "5.2b":
        run_section5_2_effective_n(seed=CONFIG.RANDOM_SEED,
                                    n_trials=5 if args.fast else 30)
