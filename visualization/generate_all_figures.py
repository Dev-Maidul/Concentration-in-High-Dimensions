"""
Generate All Figures
Creates all publication-quality figures from saved experimental results.
"""
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.global_config import CONFIG
from core.reproducibility import setup_logging, load_results
from core.plotting_utils import (plot_histogram_evolution, plot_scaling_curve,
                                plot_multi_distribution_comparison, plot_phase_transition,
                                save_figure, setup_plotting_style)


def generate_norm_concentration_figures(results_dir: str, output_dir: str):
    """Generate all norm concentration figures."""
    logger = logging.getLogger(__name__)
    logger.info("\n=== Generating Norm Concentration Figures ===")
    
    results = load_results(os.path.join(results_dir, 'norm_concentration_results.pkl'))
    
    dimensions = np.array(results['config']['dimensions'])
    distributions = results['config']['distributions']
    
    # Figure NC-1: Norm histograms for selected dimensions
    logger.info("Generating Figure NC-1: Norm histogram evolution...")
    for dist in distributions:
        hist_data = {}
        for d in [2, 50, 500, 2000]:
            if d in results['data'][dist] and results['data'][dist][d]['all_norms'] is not None:
                hist_data[d] = results['data'][dist][d]['all_norms']
        
        if hist_data:
            plot_histogram_evolution(
                hist_data,
                list(hist_data.keys()),
                xlabel="Norm",
                title=f"Norm Distribution Evolution - {dist.capitalize()}",
                filepath=os.path.join(output_dir, f"NC1_norm_histogram_{dist}")
            )
    
    # Figure NC-2: Mean norm vs dimension
    logger.info("Generating Figure NC-2: Mean norm vs dimension...")
    mean_norms = {}
    errors = {}
    for dist in distributions:
        means = []
        stds = []
        for d in dimensions:
            if d in results['data'][dist]:
                means.append(results['data'][dist][d]['mean_norm']['mean'])
                stds.append(results['data'][dist][d]['mean_norm']['std'])
            else:
                means.append(np.nan)
                stds.append(np.nan)
        mean_norms[dist] = np.array(means)
        errors[dist] = np.array(stds)
    
    plot_multi_distribution_comparison(
        dimensions, mean_norms, errors,
        xlabel="Dimension",
        ylabel="Mean Norm",
        title="Mean Norm vs Dimension",
        logx=True,
        filepath=os.path.join(output_dir, "NC2_mean_norm_vs_dimension")
    )
    
    # Figure NC-3: Relative variance vs dimension (log-log)
    logger.info("Generating Figure NC-3: Relative variance scaling...")
    rel_vars = {}
    errors = {}
    for dist in distributions:
        vars = []
        stds = []
        for d in dimensions:
            if d in results['data'][dist]:
                vars.append(results['data'][dist][d]['relative_variance']['mean'])
                stds.append(results['data'][dist][d]['relative_variance']['std'])
            else:
                vars.append(np.nan)
                stds.append(np.nan)
        rel_vars[dist] = np.array(vars)
        errors[dist] = np.array(stds)
    
    plot_multi_distribution_comparison(
        dimensions, rel_vars, errors,
        xlabel="Dimension",
        ylabel="Relative Variance (std/mean)",
        title="Relative Variance vs Dimension",
        logx=True,
        logy=True,
        filepath=os.path.join(output_dir, "NC3_relative_variance_scaling")
    )
    
    # Figure NC-4: Shell thickness ratio
    logger.info("Generating Figure NC-4: Shell thickness ratio...")
    shell_thickness = {}
    for dist in distributions:
        thicknesses = []
        for d in dimensions:
            if d in results['data'][dist]:
                thicknesses.append(results['data'][dist][d]['shell_thickness']['mean'])
            else:
                thicknesses.append(np.nan)
        shell_thickness[dist] = np.array(thicknesses)
    
    plot_multi_distribution_comparison(
        dimensions, shell_thickness,
        xlabel="Dimension",
        ylabel="Shell Thickness Ratio",
        title="Shell Thickness vs Dimension",
        logx=True,
        filepath=os.path.join(output_dir, "NC4_shell_thickness")
    )
    
    logger.info("Norm concentration figures completed.")


def generate_distance_geometry_figures(results_dir: str, output_dir: str):
    """Generate all distance geometry figures."""
    logger = logging.getLogger(__name__)
    logger.info("\n=== Generating Distance Geometry Figures ===")
    
    results = load_results(os.path.join(results_dir, 'distance_geometry_results.pkl'))
    
    dimensions = np.array(results['config']['dimensions'])
    distributions = results['config']['distributions']
    
    # Figure DG-1: Distance histogram evolution
    logger.info("Generating Figure DG-1: Distance histogram evolution...")
    dist_key = distributions[0]  # Use first distribution
    hist_data = {}
    for d in [2, 50, 500]:
        if d in results['data'][dist_key] and 'distance_distribution' in results['data'][dist_key][d]:
            hist_data[d] = results['data'][dist_key][d]['distance_distribution']
    
    if hist_data:
        plot_histogram_evolution(
            hist_data,
            list(hist_data.keys()),
            xlabel="Distance",
            title="Distance Distribution Evolution",
            filepath=os.path.join(output_dir, "DG1_distance_histogram_evolution")
        )
    
    # Figure DG-2: Relative contrast vs dimension
    logger.info("Generating Figure DG-2: Relative contrast...")
    contrasts = {}
    for dist in distributions:
        contrast_vals = []
        for d in dimensions:
            if d in results['data'][dist]:
                contrast_vals.append(results['data'][dist][d]['relative_contrast']['mean'])
            else:
                contrast_vals.append(np.nan)
        contrasts[dist] = np.array(contrast_vals)
    
    plot_multi_distribution_comparison(
        dimensions, contrasts,
        xlabel="Dimension",
        ylabel="Relative Contrast",
        title="Distance Relative Contrast vs Dimension",
        logx=True,
        logy=True,
        filepath=os.path.join(output_dir, "DG2_relative_contrast")
    )
    
    # Figure DG-3: NN ratio vs dimension
    logger.info("Generating Figure DG-3: NN ratio...")
    nn_ratios = {}
    for dist in distributions:
        ratios = []
        for d in dimensions:
            if d in results['data'][dist]:
                ratios.append(results['data'][dist][d]['nn_ratio']['mean'])
            else:
                ratios.append(np.nan)
        nn_ratios[dist] = np.array(ratios)
    
    plot_multi_distribution_comparison(
        dimensions, nn_ratios,
        xlabel="Dimension",
        ylabel="NN Distance / Mean Distance",
        title="Nearest Neighbor Ratio vs Dimension",
        logx=True,
        filepath=os.path.join(output_dir, "DG3_nn_ratio")
    )
    
    # Figure DG-4: Cosine similarity distribution narrowing
    logger.info("Generating Figure DG-4: Cosine similarity...")
    cosine_stds = {}
    for dist in distributions:
        stds = []
        for d in dimensions:
            if d in results['data'][dist]:
                stds.append(results['data'][dist][d]['cosine_similarity_std']['mean'])
            else:
                stds.append(np.nan)
        cosine_stds[dist] = np.array(stds)
    
    plot_multi_distribution_comparison(
        dimensions, cosine_stds,
        xlabel="Dimension",
        ylabel="Cosine Similarity Std Dev",
        title="Cosine Similarity Concentration",
        logx=True,
        logy=True,
        filepath=os.path.join(output_dir, "DG4_cosine_similarity")
    )
    
    logger.info("Distance geometry figures completed.")


def generate_hubness_figures(results_dir: str, output_dir: str):
    """Generate hubness analysis figures."""
    logger = logging.getLogger(__name__)
    logger.info("\n=== Generating Hubness Figures ===")
    
    try:
        results = load_results(os.path.join(results_dir, 'hubness_analysis_results.pkl'))
    except FileNotFoundError:
        logger.warning("Hubness results not found, skipping hubness figures.")
        return
    
    dimensions = np.array(results['config']['dimensions'])
    distributions = results['config']['distributions']
    
    # Figure DG-5: Hubness skewness plot
    logger.info("Generating Figure DG-5: Hubness skewness...")
    skewnesses = {}
    ginis = {}
    
    for dist in distributions:
        skew_vals = []
        gini_vals = []
        for d in dimensions:
            if d in results['data'][dist]:
                skew_vals.append(results['data'][dist][d]['skewness']['mean'])
                gini_vals.append(results['data'][dist][d]['gini_coefficient']['mean'])
            else:
                skew_vals.append(np.nan)
                gini_vals.append(np.nan)
        skewnesses[dist] = np.array(skew_vals)
        ginis[dist] = np.array(gini_vals)
    
    # Plot skewness
    plot_multi_distribution_comparison(
        dimensions, skewnesses,
        xlabel="Dimension",
        ylabel="Neighbor Count Skewness",
        title="Hubness: Neighbor Distribution Skewness",
        logx=True,
        filepath=os.path.join(output_dir, "DG5_hubness_skewness")
    )
    
    # Plot Gini coefficient
    plot_multi_distribution_comparison(
        dimensions, ginis,
        xlabel="Dimension",
        ylabel="Gini Coefficient",
        title="Hubness: Gini Coefficient",
        logx=True,
        filepath=os.path.join(output_dir, "DG5_hubness_gini")
    )
    
    logger.info("Hubness figures completed.")


def generate_jl_projection_figures(results_dir: str, output_dir: str):
    """Generate JL projection figures."""
    logger = logging.getLogger(__name__)
    logger.info("\n=== Generating JL Projection Figures ===")
    
    try:
        results = load_results(os.path.join(results_dir, 'jl_projection_results.pkl'))
    except FileNotFoundError:
        logger.warning("JL results not found, skipping JL figures.")
        return
    
    distribution = 'gaussian'
    methods = results['config']['projection_methods']
    epsilon_values = results['config']['epsilon_values']
    
    # Figure JL-2: Failure probability vs projection dimension
    logger.info("Generating Figure JL-2: Failure probability...")
    
    for original_dim in [500, 1000]:
        if original_dim not in results['data'][distribution]:
            continue
        
        fig, axes = plt.subplots(1, len(epsilon_values), figsize=(5*len(epsilon_values), 4))
        if len(epsilon_values) == 1:
            axes = [axes]
        
        for idx, eps in enumerate(epsilon_values):
            ax = axes[idx]
            
            for method in methods:
                if method not in results['data'][distribution][original_dim]:
                    continue
                
                proj_dims = []
                failure_probs = []
                
                for k, k_results in results['data'][distribution][original_dim][method].items():
                    proj_dims.append(k)
                    failure_probs.append(k_results['failure_probabilities'][eps]['mean'])
                
                proj_dims = np.array(proj_dims)
                failure_probs = np.array(failure_probs)
                
                # Sort by projection dimension
                sort_idx = np.argsort(proj_dims)
                proj_dims = proj_dims[sort_idx]
                failure_probs = failure_probs[sort_idx]
                
                ax.plot(proj_dims, failure_probs, 'o-', label=method, markersize=6)
            
            ax.set_xlabel("Projection Dimension k")
            ax.set_ylabel(f"Failure Probability")
            ax.set_title(f"ε = {eps}, d = {original_dim}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, f"JL2_failure_probability_d{original_dim}")
        save_figure(fig, filepath)
        plt.close()
    
    logger.info("JL projection figures completed.")


def generate_spectral_figures(results_dir: str, output_dir: str):
    """Generate spectral analysis figures."""
    logger = logging.getLogger(__name__)
    logger.info("\n=== Generating Spectral Analysis Figures ===")
    
    try:
        results = load_results(os.path.join(results_dir, 'spectral_analysis_results.pkl'))
    except FileNotFoundError:
        logger.warning("Spectral results not found, skipping spectral figures.")
        return
    
    distribution = 'gaussian'
    
    # Figure SP-1: Eigenvalue histogram vs MP density
    logger.info("Generating Figure SP-1: Eigenvalue distributions...")
    
    for d in [50, 200, 1000]:
        if d not in results['fixed_sample'][distribution]:
            continue
        
        dim_results = results['fixed_sample'][distribution][d]
        
        if 'eigenvalues' not in dim_results:
            continue
        
        eigenvalues = dim_results['eigenvalues']
        x_grid = dim_results['x_grid']
        mp_density = dim_results['mp_density']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Histogram of empirical eigenvalues
        ax.hist(eigenvalues, bins=50, density=True, alpha=0.6, 
               label='Empirical', edgecolor='black')
        
        # MP theoretical density
        ax.plot(x_grid, mp_density, 'r-', linewidth=2, 
               label='Marchenko-Pastur')
        
        ax.set_xlabel("Eigenvalue")
        ax.set_ylabel("Density")
        ax.set_title(f"Eigenvalue Distribution vs MP Theory (d={d}, γ={dim_results['gamma']:.3f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        filepath = os.path.join(output_dir, f"SP1_eigenvalue_dist_d{d}")
        save_figure(fig, filepath)
        plt.close()
    
    # Figure SP-2: Spectral convergence vs dimension
    logger.info("Generating Figure SP-2: Spectral convergence...")
    
    dims = []
    w_dists = []
    w_errs = []
    
    for d in sorted(results['fixed_sample'][distribution].keys()):
        dims.append(d)
        w_dists.append(results['fixed_sample'][distribution][d]['wasserstein_distance']['mean'])
        w_errs.append(results['fixed_sample'][distribution][d]['wasserstein_distance']['std'])
    
    dims = np.array(dims)
    w_dists = np.array(w_dists)
    w_errs = np.array(w_errs)
    
    plot_scaling_curve(
        dims, w_dists, w_errs,
        xlabel="Dimension",
        ylabel="Wasserstein Distance to MP",
        title="Spectral Convergence to Marchenko-Pastur",
        logx=True,
        logy=True,
        filepath=os.path.join(output_dir, "SP2_spectral_convergence")
    )
    
    # Figure SP-3: Aspect ratio study
    logger.info("Generating Figure SP-3: Aspect ratio effects...")
    
    if 'aspect_ratio_study' in results:
        gammas = sorted(results['aspect_ratio_study'][distribution].keys())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for gamma in gammas:
            gamma_results = results['aspect_ratio_study'][distribution][gamma]
            
            if 'eigenvalues' in gamma_results:
                eigenvalues = gamma_results['eigenvalues']
                x_grid = gamma_results['x_grid']
                mp_density = gamma_results['mp_density']
                
                # Plot MP density for this gamma
                ax.plot(x_grid, mp_density, linewidth=2, label=f'γ = {gamma}')
        
        ax.set_xlabel("Eigenvalue")
        ax.set_ylabel("Density")
        ax.set_title("Marchenko-Pastur Density for Different Aspect Ratios")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        filepath = os.path.join(output_dir, "SP3_aspect_ratio_study")
        save_figure(fig, filepath)
        plt.close()
    
    # Figure SP-4: PCA stability
    logger.info("Generating Figure SP-4: PCA stability...")
    
    if 'pca_stability' in results:
        for d in results['pca_stability'][distribution].keys():
            noise_levels = sorted(results['pca_stability'][distribution][d].keys())
            
            overlaps = []
            var_changes = []
            
            for noise in noise_levels:
                overlaps.append(results['pca_stability'][distribution][d][noise]['mean_overlap']['mean'])
                var_changes.append(results['pca_stability'][distribution][d][noise]['explained_variance_change']['mean'])
            
            noise_levels = np.array(noise_levels)
            overlaps = np.array(overlaps)
            var_changes = np.array(var_changes)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            ax1.plot(noise_levels, overlaps, 'o-', linewidth=2, markersize=8)
            ax1.set_xlabel("Noise Level")
            ax1.set_ylabel("Mean Eigenvector Overlap")
            ax1.set_title(f"PCA Stability: Eigenvector Overlap (d={d})")
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(noise_levels, var_changes, 'o-', linewidth=2, markersize=8, color='orange')
            ax2.set_xlabel("Noise Level")
            ax2.set_ylabel("Explained Variance Change")
            ax2.set_title(f"PCA Stability: Variance Change (d={d})")
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            filepath = os.path.join(output_dir, f"SP4_pca_stability_d{d}")
            save_figure(fig, filepath)
            plt.close()
    
    logger.info("Spectral analysis figures completed.")


def generate_cross_analysis_figures(results_dir: str, output_dir: str):
    """Generate cross-phenomenon comparison figures."""
    logger = logging.getLogger(__name__)
    logger.info("\n=== Generating Cross-Analysis Figures ===")
    
    try:
        results = load_results(os.path.join(results_dir, 'scaling_comparison_results.pkl'))
    except FileNotFoundError:
        logger.warning("Scaling comparison results not found, skipping cross-analysis figures.")
        return
    
    dimensions = np.array(results['config']['dimensions'])
    
    # Figure CP-1: Unified scaling law plot
    logger.info("Generating Figure CP-1: Unified scaling laws...")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for phenomenon, data in results['phenomena'].items():
        if 'dimensions' in data:
            dims = data['dimensions']
            values = data['values']
        else:
            dims = dimensions
            values = data['values']
        
        # Remove NaN values
        valid_mask = ~np.isnan(values)
        dims = dims[valid_mask]
        values = values[valid_mask]
        
        if len(dims) == 0:
            continue
        
        # Normalize to [0, 1] for comparison
        values_norm = (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-10)
        
        label = phenomenon.replace('_', ' ').title()
        ax.plot(dims, values_norm, 'o-', label=label, markersize=6, linewidth=2)
    
    ax.set_xlabel("Dimension", fontsize=12)
    ax.set_ylabel("Normalized Metric Value", fontsize=12)
    ax.set_title("Unified Scaling Laws: Cross-Phenomenon Comparison", fontsize=13)
    ax.set_xscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "CP1_unified_scaling_laws")
    save_figure(fig, filepath)
    plt.close()
    
    logger.info("Cross-analysis figures completed.")


def generate_all_figures():
    """Generate all figures from experimental results."""
    setup_plotting_style()
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("GENERATING ALL PUBLICATION FIGURES")
    logger.info("=" * 80)
    
    results_dir = CONFIG.RESULTS_RAW
    output_dir = CONFIG.RESULTS_FIGURES
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        generate_norm_concentration_figures(results_dir, output_dir)
        generate_distance_geometry_figures(results_dir, output_dir)
        generate_hubness_figures(results_dir, output_dir)
        generate_jl_projection_figures(results_dir, output_dir)
        generate_spectral_figures(results_dir, output_dir)
        generate_cross_analysis_figures(results_dir, output_dir)
        
        logger.info("\n" + "=" * 80)
        logger.info("ALL FIGURES GENERATED SUCCESSFULLY")
        logger.info(f"Output directory: {output_dir}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error generating figures: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    generate_all_figures()
