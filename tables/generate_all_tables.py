"""
Generate All Tables
Creates publication-ready tables from experimental results.
"""
import numpy as np
import pandas as pd
import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.global_config import CONFIG
from core.reproducibility import setup_logging, load_results


def format_value_with_ci(mean, std, decimals=3):
    """Format value with confidence interval."""
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def generate_norm_concentration_table(results_dir: str, output_dir: str):
    """Generate Table NC-1: Power-law fit coefficients for variance decay."""
    logger = logging.getLogger(__name__)
    logger.info("\nGenerating Table NC-1: Power-law fits...")
    
    results = load_results(os.path.join(results_dir, 'norm_concentration_results.pkl'))
    
    from core.scaling_analysis import fit_power_law
    
    dimensions = np.array(results['config']['dimensions'])
    distributions = results['config']['distributions']
    
    table_data = []
    
    for dist in distributions:
        rel_vars = []
        for d in dimensions:
            if d in results['data'][dist]:
                rel_vars.append(results['data'][dist][d]['relative_variance']['mean'])
            else:
                rel_vars.append(np.nan)
        
        rel_vars = np.array(rel_vars)
        valid_mask = ~np.isnan(rel_vars)
        
        if np.sum(valid_mask) > 2:
            fit = fit_power_law(dimensions[valid_mask], rel_vars[valid_mask])
            
            table_data.append({
                'Distribution': dist.capitalize(),
                'Coefficient (a)': f"{fit['coefficient']:.4e}",
                'Exponent (b)': f"{fit['exponent']:.4f}",
                'Std Error': f"{fit['std_error']:.4f}",
                'R²': f"{fit['r_squared']:.4f}"
            })
    
    df = pd.DataFrame(table_data)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, 'NC1_powerlaw_fits.csv')
    df.to_csv(csv_path, index=False)
    
    # Save as LaTeX
    latex_path = os.path.join(output_dir, 'NC1_powerlaw_fits.tex')
    with open(latex_path, 'w') as f:
        f.write(df.to_latex(index=False, escape=False, 
                           caption="Power-law fit coefficients for norm variance decay: $\sigma_{\\text{rel}} = a \\cdot d^b$"))
    
    logger.info(f"  Saved to: {csv_path}")
    logger.info(f"  Saved to: {latex_path}")
    
    return df


def generate_hubness_table(results_dir: str, output_dir: str):
    """Generate Table DG-1: Hubness statistics."""
    logger = logging.getLogger(__name__)
    logger.info("\nGenerating Table DG-1: Hubness statistics...")
    
    try:
        results = load_results(os.path.join(results_dir, 'hubness_analysis_results.pkl'))
    except FileNotFoundError:
        logger.warning("Hubness results not found, skipping table.")
        return None
    
    distributions = results['config']['distributions']
    selected_dims = [10, 50, 100, 500, 1000]
    
    table_data = []
    
    for dist in distributions:
        for d in selected_dims:
            if d not in results['data'][dist]:
                continue
            
            dim_results = results['data'][dist][d]
            
            table_data.append({
                'Distribution': dist.capitalize(),
                'Dimension': d,
                'Skewness': format_value_with_ci(
                    dim_results['skewness']['mean'],
                    dim_results['skewness']['std'],
                    decimals=3
                ),
                'Gini Coefficient': format_value_with_ci(
                    dim_results['gini_coefficient']['mean'],
                    dim_results['gini_coefficient']['std'],
                    decimals=4
                ),
                'Max Neighbor Count': f"{dim_results['max_neighbor_count']['mean']:.1f}"
            })
    
    df = pd.DataFrame(table_data)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, 'DG1_hubness_statistics.csv')
    df.to_csv(csv_path, index=False)
    
    # Save as LaTeX
    latex_path = os.path.join(output_dir, 'DG1_hubness_statistics.tex')
    with open(latex_path, 'w') as f:
        f.write(df.to_latex(index=False, escape=False,
                           caption="Hubness statistics across distributions and dimensions"))
    
    logger.info(f"  Saved to: {csv_path}")
    logger.info(f"  Saved to: {latex_path}")
    
    return df


def generate_jl_threshold_table(results_dir: str, output_dir: str):
    """Generate Table JL-1: Empirical projection dimension thresholds."""
    logger = logging.getLogger(__name__)
    logger.info("\nGenerating Table JL-1: JL projection thresholds...")
    
    try:
        results = load_results(os.path.join(results_dir, 'jl_projection_results.pkl'))
    except FileNotFoundError:
        logger.warning("JL results not found, skipping table.")
        return None
    
    distribution = 'gaussian'
    methods = results['config']['projection_methods']
    epsilon_values = results['config']['epsilon_values']
    
    table_data = []
    
    for original_dim in results['data'][distribution].keys():
        for method in methods:
            if method not in results['data'][distribution][original_dim]:
                continue
            
            row = {
                'Original Dim': original_dim,
                'Method': method.capitalize()
            }
            
            for eps in epsilon_values:
                # Find minimum k where failure probability < 0.1
                threshold_k = None
                
                proj_dims = sorted(results['data'][distribution][original_dim][method].keys())
                
                for k in proj_dims:
                    failure_prob = results['data'][distribution][original_dim][method][k]['failure_probabilities'][eps]['mean']
                    
                    if failure_prob < 0.1:  # 10% failure threshold
                        threshold_k = k
                        break
                
                if threshold_k:
                    ratio = threshold_k / original_dim
                    row[f'k (ε={eps})'] = f"{threshold_k} ({ratio:.2f}d)"
                else:
                    row[f'k (ε={eps})'] = "N/A"
            
            table_data.append(row)
    
    df = pd.DataFrame(table_data)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, 'JL1_projection_thresholds.csv')
    df.to_csv(csv_path, index=False)
    
    # Save as LaTeX
    latex_path = os.path.join(output_dir, 'JL1_projection_thresholds.tex')
    with open(latex_path, 'w') as f:
        f.write(df.to_latex(index=False, escape=False,
                           caption="Empirical JL projection dimension thresholds for 10\\% failure rate"))
    
    logger.info(f"  Saved to: {csv_path}")
    logger.info(f"  Saved to: {latex_path}")
    
    return df


def generate_spectral_convergence_table(results_dir: str, output_dir: str):
    """Generate Table SP-1: Wasserstein distances."""
    logger = logging.getLogger(__name__)
    logger.info("\nGenerating Table SP-1: Spectral convergence...")
    
    try:
        results = load_results(os.path.join(results_dir, 'spectral_analysis_results.pkl'))
    except FileNotFoundError:
        logger.warning("Spectral results not found, skipping table.")
        return None
    
    distribution = 'gaussian'
    
    table_data = []
    
    for d in sorted(results['fixed_sample'][distribution].keys()):
        dim_results = results['fixed_sample'][distribution][d]
        
        table_data.append({
            'Dimension': d,
            'Aspect Ratio (γ)': f"{dim_results['gamma']:.4f}",
            'Wasserstein Distance': format_value_with_ci(
                dim_results['wasserstein_distance']['mean'],
                dim_results['wasserstein_distance']['std'],
                decimals=4
            ),
            'Largest Eigenvalue': format_value_with_ci(
                dim_results['largest_eigenvalue']['mean'],
                dim_results['largest_eigenvalue']['std'],
                decimals=3
            )
        })
    
    df = pd.DataFrame(table_data)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, 'SP1_spectral_convergence.csv')
    df.to_csv(csv_path, index=False)
    
    # Save as LaTeX
    latex_path = os.path.join(output_dir, 'SP1_spectral_convergence.tex')
    with open(latex_path, 'w') as f:
        f.write(df.to_latex(index=False, escape=False,
                           caption="Wasserstein distance between empirical and Marchenko-Pastur distributions"))
    
    logger.info(f"  Saved to: {csv_path}")
    logger.info(f"  Saved to: {latex_path}")
    
    return df


def generate_scaling_laws_table(results_dir: str, output_dir: str):
    """Generate Table CP-1: Unified scaling law coefficients."""
    logger = logging.getLogger(__name__)
    logger.info("\nGenerating Table CP-1: Scaling law comparison...")
    
    try:
        results = load_results(os.path.join(results_dir, 'scaling_comparison_results.pkl'))
    except FileNotFoundError:
        logger.warning("Scaling comparison results not found, skipping table.")
        return None
    
    table_data = []
    
    for metric, fit_info in results['unified_fits'].items():
        row = {
            'Phenomenon': metric,
            'Best Fit': fit_info['best_fit'].replace('_', ' ').title(),
        }
        
        if fit_info['best_fit'] == 'power_law':
            row['Coefficient'] = f"{fit_info['coefficient']:.4e}"
            row['Exponent'] = f"{fit_info['exponent']:.4f} ± {fit_info['std_error']:.4f}"
            row['R²'] = f"{fit_info['r_squared']:.4f}"
        else:
            row['Coefficient'] = f"{fit_info['coefficient']:.4e}"
            row['Decay Rate'] = f"{fit_info['decay_rate']:.4f}"
            row['R²'] = f"{fit_info['r_squared']:.4f}"
        
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, 'CP1_scaling_laws.csv')
    df.to_csv(csv_path, index=False)
    
    # Save as LaTeX
    latex_path = os.path.join(output_dir, 'CP1_scaling_laws.tex')
    with open(latex_path, 'w') as f:
        f.write(df.to_latex(index=False, escape=False,
                           caption="Unified scaling law comparison across all phenomena"))
    
    logger.info(f"  Saved to: {csv_path}")
    logger.info(f"  Saved to: {latex_path}")
    
    return df


def generate_all_tables():
    """Generate all tables from experimental results."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("GENERATING ALL PUBLICATION TABLES")
    logger.info("=" * 80)
    
    results_dir = CONFIG.RESULTS_RAW
    output_dir = CONFIG.RESULTS_TABLES
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        generate_norm_concentration_table(results_dir, output_dir)
        generate_hubness_table(results_dir, output_dir)
        generate_jl_threshold_table(results_dir, output_dir)
        generate_spectral_convergence_table(results_dir, output_dir)
        generate_scaling_laws_table(results_dir, output_dir)
        
        logger.info("\n" + "=" * 80)
        logger.info("ALL TABLES GENERATED SUCCESSFULLY")
        logger.info(f"Output directory: {output_dir}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error generating tables: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    generate_all_tables()
