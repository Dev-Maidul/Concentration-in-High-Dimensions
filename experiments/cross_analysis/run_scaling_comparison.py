"""
Cross-Phenomenon Scaling Laws Analysis
Compares scaling behavior across all concentration phenomena.
"""
import numpy as np
import logging
from typing import Dict
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config.global_config import CONFIG
from core.reproducibility import setup_logging, save_results, load_results
from core.scaling_analysis import fit_power_law, compare_scaling_laws


def run_scaling_comparison() -> Dict:
    """
    Load results from all experiments and compare scaling laws.
    
    Returns
    -------
    dict
        Unified scaling analysis results
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Cross-Phenomenon Scaling Analysis")
    
    # Load results from all experiments
    results_dir = CONFIG.RESULTS_RAW
    
    try:
        norm_results = load_results(os.path.join(results_dir, 'norm_concentration_results.pkl'))
        distance_results = load_results(os.path.join(results_dir, 'distance_geometry_results.pkl'))
        jl_results = load_results(os.path.join(results_dir, 'jl_projection_results.pkl'))
        spectral_results = load_results(os.path.join(results_dir, 'spectral_analysis_results.pkl'))
    except FileNotFoundError as e:
        logger.error(f"Missing experimental results: {e}")
        logger.error("Please run individual experiments first.")
        return None
    
    # Extract dimensions (use norm experiment as reference)
    dimensions = np.array(norm_results['config']['dimensions'])
    
    # Focus on Gaussian distribution for cross-comparison
    distribution = 'gaussian'
    
    scaling_results = {
        'config': {
            'dimensions': dimensions.tolist(),
            'distribution': distribution
        },
        'phenomena': {},
        'unified_fits': {}
    }
    
    # 1. Norm variance scaling
    logger.info("\n1. Analyzing norm variance scaling...")
    norm_variance = []
    for d in dimensions:
        if d in norm_results['data'][distribution]:
            rel_var = norm_results['data'][distribution][d]['relative_variance']['mean']
            norm_variance.append(rel_var)
        else:
            norm_variance.append(np.nan)
    
    norm_variance = np.array(norm_variance)
    valid_mask = ~np.isnan(norm_variance)
    
    norm_fit = fit_power_law(dimensions[valid_mask], norm_variance[valid_mask])
    
    scaling_results['phenomena']['norm_variance'] = {
        'values': norm_variance,
        'fit': norm_fit
    }
    
    # 2. Distance variance scaling
    logger.info("2. Analyzing distance variance scaling...")
    distance_variance = []
    for d in dimensions:
        if d in distance_results['data'][distribution]:
            rel_var = distance_results['data'][distribution][d]['relative_contrast']['mean']
            distance_variance.append(rel_var)
        else:
            distance_variance.append(np.nan)
    
    distance_variance = np.array(distance_variance)
    valid_mask = ~np.isnan(distance_variance)
    
    distance_fit = fit_power_law(dimensions[valid_mask], distance_variance[valid_mask])
    
    scaling_results['phenomena']['distance_variance'] = {
        'values': distance_variance,
        'fit': distance_fit
    }
    
    # 3. JL distortion scaling (if available)
    logger.info("3. Analyzing JL distortion scaling...")
    if distribution in jl_results['data']:
        # Extract JL distortion for k = d/2
        jl_dimensions = []
        jl_distortions = []
        
        for d in jl_results['data'][distribution].keys():
            if 'gaussian' in jl_results['data'][distribution][d]:
                # Find projection dimension closest to d/2
                k_values = jl_results['data'][distribution][d]['gaussian'].keys()
                target_k = d / 2
                closest_k = min(k_values, key=lambda k: abs(k - target_k))
                
                mean_dist = jl_results['data'][distribution][d]['gaussian'][closest_k]['mean_distortion']['mean']
                jl_dimensions.append(d)
                jl_distortions.append(abs(mean_dist - 1.0))  # Distortion from 1
        
        jl_dimensions = np.array(jl_dimensions)
        jl_distortions = np.array(jl_distortions)
        
        if len(jl_dimensions) > 0:
            jl_fit = fit_power_law(jl_dimensions, jl_distortions)
            
            scaling_results['phenomena']['jl_distortion'] = {
                'dimensions': jl_dimensions,
                'values': jl_distortions,
                'fit': jl_fit
            }
    
    # 4. Spectral convergence scaling
    logger.info("4. Analyzing spectral convergence scaling...")
    spectral_dimensions = []
    spectral_convergence = []
    
    for d in spectral_results['fixed_sample'][distribution].keys():
        w_dist = spectral_results['fixed_sample'][distribution][d]['wasserstein_distance']['mean']
        spectral_dimensions.append(d)
        spectral_convergence.append(w_dist)
    
    spectral_dimensions = np.array(spectral_dimensions)
    spectral_convergence = np.array(spectral_convergence)
    
    spectral_fit = fit_power_law(spectral_dimensions, spectral_convergence)
    
    scaling_results['phenomena']['spectral_convergence'] = {
        'dimensions': spectral_dimensions,
        'values': spectral_convergence,
        'fit': spectral_fit
    }
    
    # Unified comparison
    logger.info("\n5. Computing unified scaling comparison...")
    
    all_metrics = {
        'Norm Variance': norm_variance[valid_mask],
        'Distance Contrast': distance_variance[valid_mask]
    }
    
    if 'jl_distortion' in scaling_results['phenomena']:
        # Interpolate JL distortion to common dimensions
        jl_interp = np.interp(dimensions[valid_mask], 
                             jl_dimensions, 
                             jl_distortions, 
                             left=np.nan, right=np.nan)
        all_metrics['JL Distortion'] = jl_interp
    
    # Interpolate spectral convergence
    spectral_interp = np.interp(dimensions[valid_mask],
                               spectral_dimensions,
                               spectral_convergence,
                               left=np.nan, right=np.nan)
    all_metrics['Spectral Convergence'] = spectral_interp
    
    unified_fits = compare_scaling_laws(dimensions[valid_mask], all_metrics)
    
    scaling_results['unified_fits'] = unified_fits
    
    # Summary statistics
    logger.info("\n=== Scaling Law Summary ===")
    for metric, fit_info in unified_fits.items():
        logger.info(f"\n{metric}:")
        logger.info(f"  Best fit: {fit_info['best_fit']}")
        if fit_info['best_fit'] == 'power_law':
            logger.info(f"  Exponent: {fit_info['exponent']:.4f} ± {fit_info['std_error']:.4f}")
            logger.info(f"  R²: {fit_info['r_squared']:.4f}")
    
    # Save results
    output_path = os.path.join(CONFIG.RESULTS_RAW, 'scaling_comparison_results.pkl')
    save_results(scaling_results, output_path, format='pickle')
    
    logger.info(f"\nScaling comparison completed. Results saved to: {output_path}")
    
    return scaling_results


if __name__ == "__main__":
    results = run_scaling_comparison()
