"""
generate_prediction_table.py
------------------------------
Section 7.2: Generate the unified prediction table for practitioners.

Given a dataset with estimated intrinsic dimension d_int, this table
lets a practitioner predict:
  - Norm coefficient of variation
  - Distance relative contrast
  - Nearest-neighbor ratio
  - Hubness Gini coefficient
  - Practical recommendation

Also generates:
  - LaTeX source for the table
  - Overlay figure: synthetic curves + real embedding data points
  - Section 7 scaling hierarchy plot (updated with real embedding results)
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

PROJECT_ROOT = Path("/mnt/data2/naeem/Geometry-and-Concentration-in-High-Dimensions-main")
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results" / "raw" / "real_embeddings"
TABLES_DIR  = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Scaling law constants ──────────────────────────────────────────────────────
A_NORM, B_NORM = 0.728, -0.505
A_DIST, B_DIST = 17.35, -0.623

# Empirical NN ratio approximation fitted to synthetic data
def _nn_ratio(d):
    """Approximate NN ratio from synthetic data fit."""
    return float(1.0 - 0.41 * d ** -0.346)

# Approximate Gini from synthetic Gaussian data
def _gini_pred(d):
    """Approximate Gini coefficient from power-law fit to synthetic data."""
    if d <= 2:
        return 0.131
    # Fitted sigmoid-like curve to synthetic Gini data
    return float(0.131 + (0.849 - 0.131) * min(1.0, (d / 2000) ** 0.35))

def _recommendation(d_int: float) -> str:
    if d_int < 50:
        return "kNN reliable"
    elif d_int < 100:
        return "Monitor hubness"
    elif d_int < 200:
        return "Reduce dimension"
    elif d_int < 500:
        return "kNN unreliable"
    else:
        return "Must reduce dim"


def build_prediction_table(d_int_values=None) -> Dict:
    """
    Build the unified prediction table.

    Returns a dict with predicted values for each d_int.
    """
    if d_int_values is None:
        d_int_values = [5, 10, 20, 50, 100, 150, 200, 300, 500, 750, 1000, 2000]

    table = {}
    for d in d_int_values:
        table[d] = {
            "d_int":        d,
            "norm_cv":      float(A_NORM * d ** B_NORM),
            "dist_contrast": float(A_DIST * d ** B_DIST),
            "nn_ratio":     _nn_ratio(d),
            "hubness_gini": _gini_pred(d),
            "recommendation": _recommendation(d),
        }

    return table


def load_real_results() -> Dict:
    """Load real embedding results with intrinsic dimensions."""
    combined = RESULTS_DIR / "embedding_geometry_full.json"
    if not combined.exists():
        logger.warning("Real embedding results not found: %s", combined)
        return {}
    with open(combined) as fh:
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


def generate_latex_table(pred_table: Dict, real_data: Dict) -> str:
    """
    Generate LaTeX source for the unified prediction table (Table 7 in paper).
    Includes both synthetic predictions and real embedding data points.
    """
    header = (
        r"\begin{table}[t]" + "\n"
        r"\centering" + "\n"
        r"\caption{Unified geometric prediction table. For any dataset with "
        r"estimated intrinsic dimension $d_\text{int}$, these values predict "
        r"geometric behaviour from the empirical scaling laws. "
        r"Real embedding rows (shaded) show observed values; "
        r"deviation from prediction in parentheses.}" + "\n"
        r"\label{tab:prediction_table}" + "\n"
        r"\small" + "\n"
        r"\begin{tabular}{lrrrrrl}" + "\n"
        r"\toprule" + "\n"
        r"$d_\text{int}$ & $\sigma_\text{rel}$ & $C_\text{rel}$ & "
        r"NN Ratio & Gini $G$ & Recommendation \\" + "\n"
        r"\midrule" + "\n"
    )

    rows = ""

    # Synthetic prediction rows
    rows += r"\multicolumn{6}{l}{\textit{Synthetic predictions}} \\" + "\n"
    rows += r"\midrule" + "\n"
    for d, v in pred_table.items():
        rec = v["recommendation"].replace("dim", "dim.")
        gini_fmt = f"\\textbf{{{v['hubness_gini']:.3f}}}" if d >= 100 else f"{v['hubness_gini']:.3f}"
        rows += (
            f"{d} & "
            f"{v['norm_cv']:.4f} & "
            f"{v['dist_contrast']:.4f} & "
            f"{v['nn_ratio']:.3f} & "
            f"{gini_fmt} & "
            f"{rec} \\\\\n"
        )

    # Real embedding rows
    if real_data:
        rows += r"\midrule" + "\n"
        rows += r"\multicolumn{6}{l}{\textit{Real embeddings (observed)}} \\" + "\n"
        rows += r"\midrule" + "\n"

        # Short display names
        display_names = {
            "glove_50d":         "GloVe-50",
            "glove_100d":        "GloVe-100",
            "glove_200d":        "GloVe-200",
            "glove_300d":        "GloVe-300",
            "word2vec_300d":     "Word2Vec",
            "bert_768d":         "BERT-768",
            "cifar10_raw":       "CIFAR10-raw",
            "cifar10_resnet2048": "CIFAR10-R50",
            "mnist_raw":         "MNIST-raw",
            "mnist_pca50":       "MNIST-PCA50",
            "scrna_pbmc":        "scRNA-PBMC",
        }

        for name, result in real_data.items():
            if result.get("status") != "ok":
                continue
            d_int  = _get(result, "d_int")
            if d_int is None:
                continue

            obs_norm = _get(result, "norm", "relative_variance")
            obs_dist = _get(result, "distance", "rel_contrast")
            obs_nn   = _get(result, "distance", "nn_ratio")
            obs_gini = _get(result, "hubness", "gini")

            # Predicted at this d_int
            pred_norm = A_NORM * (d_int ** B_NORM)
            pred_dist = A_DIST * (d_int ** B_DIST)

            def _fmt_with_dev(obs, pred):
                if obs is None:
                    return "—"
                dev = 100 * (obs - pred) / pred if pred > 0 else 0
                sign = "+" if dev >= 0 else ""
                return f"{obs:.3f} ({sign}{dev:.0f}\\%)"

            label = display_names.get(name, name)
            rec   = _recommendation(d_int)

            rows += (
                f"\\rowcolor{{gray!12}} "
                f"{label} ($d_\\text{{int}}$={d_int:.0f}) & "
                f"{_fmt_with_dev(obs_norm, pred_norm)} & "
                f"{_fmt_with_dev(obs_dist, pred_dist)} & "
                f"{f'{obs_nn:.3f}' if obs_nn is not None else '—'} & "
                f"{f'{obs_gini:.3f}' if obs_gini is not None else '—'} & "
                f"{rec} \\\\\n"
            )

    footer = (
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
        r"\end{table}"
    )
    return header + rows + footer


def generate_overlay_figure(pred_table: Dict, real_data: Dict):
    """
    Generate the overlay figure: synthetic scaling law curves with
    real embedding data points plotted at their d_int values.
    This is the key visual for Section 7.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib not available, skipping figure")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Empirical Scaling Laws: Synthetic Predictions vs Real Embeddings",
                 fontsize=13, fontweight="bold", y=1.01)

    # Synthetic curves
    d_range = np.logspace(np.log10(5), np.log10(2000), 200)

    # Color and marker scheme for real datasets
    real_styles = {
        "glove_50d":          ("#9467bd", "D", "GloVe-50"),
        "glove_100d":         ("#9467bd", "s", "GloVe-100"),
        "glove_200d":         ("#9467bd", "^", "GloVe-200"),
        "glove_300d":         ("#9467bd", "o", "GloVe-300"),
        "word2vec_300d":      ("#8c564b", "D", "Word2Vec"),
        "bert_768d":          ("#e377c2", "*", "BERT-768"),
        "cifar10_raw":        ("#17becf", "D", "CIFAR10-raw"),
        "cifar10_resnet2048": ("#17becf", "s", "CIFAR10-ResNet"),
        "mnist_raw":          ("#bcbd22", "D", "MNIST-raw"),
        "mnist_pca50":        ("#bcbd22", "s", "MNIST-PCA50"),
        "scrna_pbmc":         ("#7f7f7f", "*", "scRNA-PBMC"),
    }

    for ax_idx, (ax, metric, ylabel, pred_fn, law_label) in enumerate(zip(
        axes,
        ["norm_cv", "dist_contrast", "hubness_gini"],
        [r"Relative Norm Variance $\sigma_{rel}$",
         r"Distance Contrast $C_{rel}$",
         r"Hubness Gini $G$"],
        [lambda d: A_NORM * d ** B_NORM,
         lambda d: A_DIST * d ** B_DIST,
         _gini_pred],
        [r"$0.728 \cdot d^{-0.505}$",
         r"$17.35 \cdot d^{-0.623}$",
         "Synthetic Gaussian"],
    )):
        # Synthetic curve
        y_pred = np.array([pred_fn(d) for d in d_range])
        if metric != "hubness_gini":
            ax.loglog(d_range, y_pred, "k-", lw=2, label=law_label, alpha=0.7)
            ax.set_xscale("log")
            ax.set_yscale("log")
        else:
            ax.semilogx(d_range, y_pred, "k-", lw=2, label=law_label, alpha=0.7)
            ax.axhline(0.75, color="grey", ls=":", lw=1.2, alpha=0.7,
                       label="G = 0.75 threshold")

        # Real embedding points at d_int
        for name, result in real_data.items():
            if result.get("status") != "ok":
                continue
            if name not in real_styles:
                continue

            d_int = _get(result, "d_int")
            if d_int is None:
                continue

            color, marker, label = real_styles[name]

            if metric == "norm_cv":
                obs = _get(result, "norm", "relative_variance")
            elif metric == "dist_contrast":
                obs = _get(result, "distance", "rel_contrast")
            else:
                obs = _get(result, "hubness", "gini")

            if obs is None:
                continue

            ax.scatter(d_int, obs, c=color, marker=marker, s=80,
                       zorder=5, edgecolors="k", linewidths=0.6, label=label)

        ax.set_xlabel("Intrinsic Dimension $d_{int}$", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(labelsize=8)

        # Mark d_int = 100 threshold
        ax.axvline(100, color="red", ls="--", lw=1, alpha=0.5)

    # Shared legend (deduplicated)
    handles, labels = [], []
    seen = set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                handles.append(h)
                labels.append(l)
                seen.add(l)

    fig.legend(handles, labels, loc="lower center", ncol=6,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()

    for fmt in ["pdf", "png"]:
        path = FIGURES_DIR / f"figure_scaling_law_overlay.{fmt}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        logger.info("Saved: %s", path)
    plt.close(fig)


def generate_hierarchy_figure(pred_table: Dict, real_data: Dict):
    """
    Updated Figure 6 (scaling hierarchy) including real embedding data points.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    d_range = np.logspace(np.log10(5), np.log10(2000), 200)

    # Normalise each series to start at 1.0 at d=5
    def _norm(vals):
        return vals / vals[0]

    norm_var  = _norm(A_NORM * d_range ** B_NORM)
    dist_cont = _norm(A_DIST * d_range ** B_DIST)
    spectral  = _norm(0.0138 * d_range ** -0.372)
    jl_dist   = _norm(0.440 * d_range ** -0.944)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(d_range, dist_cont, color="#2ca02c", lw=2.5,
            label=r"Distance Contrast ($d^{-0.623}$, steepest)")
    ax.plot(d_range, norm_var,  color="#1f77b4", lw=2.5,
            label=r"Norm Variance ($d^{-0.505}$)")
    ax.plot(d_range, spectral,  color="#d62728", lw=2.5,
            label=r"Spectral Convergence ($d^{-0.372}$)")
    ax.plot(d_range, jl_dist,   color="#9467bd", lw=2.5,
            label=r"JL Distortion ($d^{-0.944}$, slowest)")

    # Real embedding data points (distance contrast only, as it's the key law)
    real_colors = {
        "glove_300d": "#17becf",
        "bert_768d":  "#e377c2",
        "scrna_pbmc": "#7f7f7f",
        "mnist_raw":  "#bcbd22",
    }
    d5_dist = A_DIST * 5 ** B_DIST  # normalisation reference

    for name, color in real_colors.items():
        result = real_data.get(name, {})
        if result.get("status") != "ok":
            continue
        d_int = _get(result, "d_int")
        obs   = _get(result, "distance", "rel_contrast")
        if d_int and obs:
            obs_norm = obs / d5_dist
            ax.scatter(d_int, obs_norm, c=color, s=100, zorder=6,
                       edgecolors="k", linewidths=0.8,
                       label=f"{name.replace('_', '-')} (real)")

    ax.axvline(100, color="grey", ls=":", lw=1.2, alpha=0.7,
               label="$d_{int} = 100$ threshold")
    ax.set_xscale("log")
    ax.set_xlabel("Dimension $d$ (or $d_{int}$ for real embeddings)", fontsize=11)
    ax.set_ylabel("Normalised Metric Value", fontsize=11)
    ax.set_title("Unified Scaling Laws: Hierarchy and Real Embedding Validation",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()

    for fmt in ["pdf", "png"]:
        path = FIGURES_DIR / f"figure_hierarchy_updated.{fmt}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        logger.info("Saved: %s", path)
    plt.close(fig)


def main():
    logger.info("Generating Section 7 prediction table and figures …")

    # Build prediction table
    pred_table = build_prediction_table()

    # Load real embedding results
    real_data = load_real_results()
    if not real_data:
        logger.warning("No real embedding results found. "
                       "Run run_embedding_geometry.py first.")

    # Generate LaTeX table
    latex = generate_latex_table(pred_table, real_data)
    latex_path = TABLES_DIR / "table_prediction.tex"
    latex_path.write_text(latex)
    logger.info("LaTeX table: %s", latex_path)

    # Print plain-text preview
    print("\n" + "=" * 80)
    print("PREDICTION TABLE (synthetic rows only shown here)")
    print(f"{'d_int':>6} {'Norm CV':>10} {'Crel':>10} {'NN Ratio':>10} "
          f"{'Gini':>8} {'Recommendation'}")
    print("-" * 80)
    for d, v in pred_table.items():
        flag = " ◄" if d == 100 else ""
        print(f"{d:>6} {v['norm_cv']:>10.4f} {v['dist_contrast']:>10.4f} "
              f"{v['nn_ratio']:>10.3f} {v['hubness_gini']:>8.3f} "
              f"{v['recommendation']}{flag}")
    print("=" * 80)
    print("◄ = critical threshold (d_int ≈ 100)")

    # Generate figures
    generate_overlay_figure(pred_table, real_data)
    generate_hierarchy_figure(pred_table, real_data)

    logger.info("Section 7 outputs complete.")


if __name__ == "__main__":
    main()
