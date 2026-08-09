"""
Spectral Analysis Experiment
Analyzes eigenvalue distributions and compares with Marchenko-Pastur.
"""
import numpy as np
import logging
from typing import Dict, List
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config.global_config import CONFIG, set_seed
from core.random_generators import generate_data, generate_aspect_ratio_data
from core.spectral_methods import (compute_spectral_statistics, 
                                  compute_pca_stability)
from core.reproducibility import setup_logging, save_results
from joblib import Parallel, delayed


def run_single_spectral_trial(distribution: str, n: int, d: int, 
                             trial_idx: int) -> Dict:
    """
    Run single trial of spectral analysis.
    
    Parameters
    ----------
    distribution : str
        Distribution name
    n : int
        Number of samples
    d : int
        Dimension
    trial_idx : int
        Trial index for seeding
    
    Returns
    -------
    dict
        Spectral statistics
    """
    seed = CONFIG.RANDOM_SEED + trial_idx
    X = generate_data(distribution, n, d, seed=seed)
    
    gamma = d / n
    spectral_stats = compute_spectral_statistics(X, gamma=gamma)
    
    # For selected dimensions, save eigenvalue distribution
    if d in [50, 200, 1000]:
        eigenvalues = spectral_stats['eigenvalues']
        x_grid = spectral_stats['x_grid']
        mp_density = spectral_stats['mp_density']
    else:
        eigenvalues = None
        x_grid = None
        mp_density = None
    
    return {
        'wasserstein_distance': spectral_stats['wasserstein_distance'],
        'largest_eigenvalue': spectral_stats['largest_eigenvalue'],
        'smallest_eigenvalue': spectral_stats['smallest_eigenvalue'],
        'condition_number': spectral_stats['condition_number'],
        'gamma': gamma,
        'eigenvalues': eigenvalues,
        'x_grid': x_grid,
        'mp_density': mp_density
    }


def run_aspect_ratio_trial(distribution: str, d: int, gamma: float, 
                          trial_idx: int) -> Dict:
    """
    Run single trial for aspect ratio study.
    
    Parameters
    ----------
    distribution : str
        Distribution name
    d : int
        Dimension
    gamma : float
        Aspect ratio d/n
    trial_idx : int
        Trial index for seeding
    
    Returns
    -------
    dict
        Spectral statistics
    """
    seed = CONFIG.RANDOM_SEED + trial_idx
    X = generate_aspect_ratio_data(distribution, d, gamma, seed=seed)
    
    spectral_stats = compute_spectral_statistics(X, gamma=gamma)
    
    return {
        'wasserstein_distance': spectral_stats['wasserstein_distance'],
        'largest_eigenvalue': spectral_stats['largest_eigenvalue'],
        'condition_number': spectral_stats['condition_number'],
        'eigenvalues': spectral_stats['eigenvalues'],
        'x_grid': spectral_stats['x_grid'],
        'mp_density': spectral_stats['mp_density']
    }


def run_pca_stability_trial(distribution: str, n: int, d: int,
                           noise_level: float, trial_idx: int) -> Dict:
    """
    Run single trial of PCA stability analysis.
    
    Parameters
    ----------
    distribution : str
        Distribution name
    n : int
        Number of samples
    d : int
        Dimension
    noise_level : float
        Noise standard deviation
    trial_idx : int
        Trial index for seeding
    
    Returns
    -------
    dict
        PCA stability metrics
    """
    seed = CONFIG.RANDOM_SEED + trial_idx
    X = generate_data(distribution, n, d, seed=seed)
    
    stability_stats = compute_pca_stability(X, noise_level, 
                                           n_components=min(10, d, n),
                                           seed=seed)
    
    return stability_stats


def run_spectral_analysis_experiment(distributions: List[str] = None,
                                    dimensions: List[int] = None,
                                    aspect_ratios: List[float] = None,
                                    n_samples: int = None,
                                    n_trials: int = None,
                                    noise_levels: List[float] = None,
                                    n_jobs: int = None) -> Dict:
    """
    Run complete spectral analysis experiment.
    
    Parameters
    ----------
    distributions : list, optional
        List of distributions to test
    dimensions : list, optional
        List of dimensions to test
    aspect_ratios : list, optional
        Aspect ratios for gamma study
    n_samples : int, optional
        Number of samples per trial
    n_trials : int, optional
        Number of independent trials
    noise_levels : list, optional
        Noise levels for PCA stability
    n_jobs : int, optional
        Number of parallel jobs
    
    Returns
    -------
    dict
        Experimental results
    """
    if distributions is None:
        distributions = ['gaussian']  # Focus on Gaussian for spectral
    if dimensions is None:
        dimensions = [50, 100, 200, 500, 1000]
    if aspect_ratios is None:
        aspect_ratios = CONFIG.ASPECT_RATIOS
    if n_samples is None:
        n_samples = CONFIG.FIXED_SAMPLE_SIZE
    if n_trials is None:
        n_trials = CONFIG.N_TRIALS
    if noise_levels is None:
        noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]
    if n_jobs is None:
        n_jobs = CONFIG.N_JOBS
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting Spectral Analysis Experiment")
    logger.info(f"Distributions: {distributions}")
    logger.info(f"Dimensions: {dimensions}")
    logger.info(f"Aspect ratios: {aspect_ratios}")
    
    results = {
        'config': {
            'distributions': distributions,
            'dimensions': dimensions,
            'aspect_ratios': aspect_ratios,
            'n_samples': n_samples,
            'n_trials': n_trials,
            'noise_levels': noise_levels
        },
        'fixed_sample': {},
        'aspect_ratio_study': {},
        'pca_stability': {}
    }
    
    # Part 1: Fixed sample size
    logger.info("\n=== Part 1: Fixed Sample Size ===")
    for distribution in distributions:
        logger.info(f"\nProcessing distribution: {distribution}")
        results['fixed_sample'][distribution] = {}
        
        for d in tqdm(dimensions, desc=f"{distribution}"):
            trial_results = Parallel(n_jobs=n_jobs)(
                delayed(run_single_spectral_trial)(distribution, n_samples, d, trial_idx)
                for trial_idx in range(n_trials)
            )
            
            # Aggregate results
            w_distances = [r['wasserstein_distance'] for r in trial_results]
            largest_eigs = [r['largest_eigenvalue'] for r in trial_results]
            cond_numbers = [r['condition_number'] for r in trial_results if np.isfinite(r['condition_number'])]
            
            dim_results = {
                'wasserstein_distance': {
                    'mean': np.mean(w_distances),
                    'std': np.std(w_distances),
                    'values': w_distances
                },
                'largest_eigenvalue': {
                    'mean': np.mean(largest_eigs),
                    'std': np.std(largest_eigs)
                },
                'condition_number': {
                    'mean': np.mean(cond_numbers) if cond_numbers else np.nan,
                    'std': np.std(cond_numbers) if cond_numbers else np.nan
                },
                'gamma': d / n_samples
            }
            
            # Collect eigenvalue distributions for selected dimensions
            if d in [50, 200, 1000]:
                for r in trial_results:
                    if r['eigenvalues'] is not None:
                        dim_results['eigenvalues'] = r['eigenvalues']
                        dim_results['x_grid'] = r['x_grid']
                        dim_results['mp_density'] = r['mp_density']
                        break
            
            results['fixed_sample'][distribution][d] = dim_results
    
    # Part 2: Aspect ratio study
    logger.info("\n=== Part 2: Aspect Ratio Study ===")
    test_dimension = 500
    for distribution in distributions:
        logger.info(f"\nProcessing distribution: {distribution}")
        results['aspect_ratio_study'][distribution] = {}
        
        for gamma in tqdm(aspect_ratios, desc=f"{distribution}"):
            trial_results = Parallel(n_jobs=n_jobs)(
                delayed(run_aspect_ratio_trial)(distribution, test_dimension, gamma, trial_idx)
                for trial_idx in range(n_trials)
            )
            
            # Aggregate results
            w_distances = [r['wasserstein_distance'] for r in trial_results]
            largest_eigs = [r['largest_eigenvalue'] for r in trial_results]
            
            gamma_results = {
                'wasserstein_distance': {
                    'mean': np.mean(w_distances),
                    'std': np.std(w_distances)
                },
                'largest_eigenvalue': {
                    'mean': np.mean(largest_eigs),
                    'std': np.std(largest_eigs)
                },
                'eigenvalues': trial_results[0]['eigenvalues'],
                'x_grid': trial_results[0]['x_grid'],
                'mp_density': trial_results[0]['mp_density']
            }
            
            results['aspect_ratio_study'][distribution][gamma] = gamma_results
    
    # Part 3: PCA stability
    logger.info("\n=== Part 3: PCA Stability ===")
    test_dims = [100, 500]
    for distribution in distributions:
        logger.info(f"\nProcessing distribution: {distribution}")
        results['pca_stability'][distribution] = {}
        
        for d in test_dims:
            results['pca_stability'][distribution][d] = {}
            
            for noise_level in tqdm(noise_levels, desc=f"{distribution}, d={d}"):
                trial_results = Parallel(n_jobs=n_jobs)(
                    delayed(run_pca_stability_trial)(distribution, n_samples, d, 
                                                    noise_level, trial_idx)
                    for trial_idx in range(n_trials)
                )
                
                # Aggregate results
                mean_overlaps = [r['mean_overlap'] for r in trial_results]
                var_changes = [r['explained_variance_change'] for r in trial_results]
                
                noise_results = {
                    'mean_overlap': {
                        'mean': np.mean(mean_overlaps),
                        'std': np.std(mean_overlaps),
                        'values': mean_overlaps
                    },
                    'explained_variance_change': {
                        'mean': np.mean(var_changes),
                        'std': np.std(var_changes),
                        'values': var_changes
                    }
                }
                
                results['pca_stability'][distribution][d][noise_level] = noise_results
    
    # Save results
    output_path = os.path.join(CONFIG.RESULTS_RAW, 'spectral_analysis_results.pkl')
    save_results(results, output_path, format='pickle')
    
    logger.info(f"\nSpectral analysis completed. Results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    set_seed()
    results = run_spectral_analysis_experiment()
