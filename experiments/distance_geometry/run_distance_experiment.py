"""
Distance Geometry Experiment
Analyzes pairwise distances, nearest neighbors, and cosine similarity.
"""
import numpy as np
import logging
from typing import Dict, List
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config.global_config import CONFIG, set_seed
from core.random_generators import generate_data
from core.metrics import (compute_distance_statistics, 
                          compute_nearest_neighbor_statistics,
                          compute_cosine_similarity)
from core.reproducibility import setup_logging, save_results
from joblib import Parallel, delayed


def run_single_trial(distribution: str, n: int, d: int, trial_idx: int,
                    subsample: int = None) -> Dict:
    """
    Run single trial of distance geometry experiment.
    
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
    subsample : int, optional
        Subsample size for distance computation
    
    Returns
    -------
    dict
        Distance statistics for this trial
    """
    seed = CONFIG.RANDOM_SEED + trial_idx
    X = generate_data(distribution, n, d, seed=seed)
    
    # Distance statistics
    dist_stats = compute_distance_statistics(X, subsample=subsample)
    
    # Nearest neighbor statistics
    nn_stats = compute_nearest_neighbor_statistics(X, k=1)
    
    # Cosine similarity
    cosine_sims = compute_cosine_similarity(X, subsample=subsample)
    
    # Compute NN ratio
    nn_ratio = nn_stats['mean_nn_distance'] / dist_stats['mean_distance']
    
    results = {
        **dist_stats,
        'nn_distance': nn_stats['mean_nn_distance'],
        'nn_ratio': nn_ratio,
        'cosine_similarity_mean': np.mean(cosine_sims),
        'cosine_similarity_std': np.std(cosine_sims),
    }
    
    # For selected dimensions, save full distributions
    if d in [2, 50, 500]:
        from scipy.spatial.distance import pdist
        # Get subsample for histogram
        if subsample and subsample < X.shape[0]:
            idx = np.random.choice(X.shape[0], min(500, subsample), replace=False)
            X_sub = X[idx]
        else:
            X_sub = X[:500]
        
        distances = pdist(X_sub)
        results['distance_distribution'] = distances
        results['cosine_distribution'] = cosine_sims[:5000]  # Limit size
    
    return results


def run_distance_geometry_experiment(distributions: List[str] = None,
                                    dimensions: List[int] = None,
                                    n_samples: int = None,
                                    n_trials: int = None,
                                    subsample: int = None,
                                    n_jobs: int = None) -> Dict:
    """
    Run complete distance geometry experiment.
    
    Parameters
    ----------
    distributions : list, optional
        List of distributions to test
    dimensions : list, optional
        List of dimensions to test
    n_samples : int, optional
        Number of samples per trial
    n_trials : int, optional
        Number of independent trials
    subsample : int, optional
        Subsample size for distance computation
    n_jobs : int, optional
        Number of parallel jobs
    
    Returns
    -------
    dict
        Experimental results
    """
    if distributions is None:
        distributions = CONFIG.DISTRIBUTIONS
    if dimensions is None:
        dimensions = CONFIG.DIMENSIONS
    if n_samples is None:
        n_samples = CONFIG.FIXED_SAMPLE_SIZE
    if n_trials is None:
        n_trials = CONFIG.N_TRIALS
    if subsample is None:
        subsample = CONFIG.DISTANCE_SUBSAMPLE
    if n_jobs is None:
        n_jobs = CONFIG.N_JOBS
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting Distance Geometry Experiment")
    logger.info(f"Distributions: {distributions}")
    logger.info(f"Dimensions: {dimensions}")
    logger.info(f"Samples: {n_samples}, Trials: {n_trials}, Subsample: {subsample}")
    
    results = {
        'config': {
            'distributions': distributions,
            'dimensions': dimensions,
            'n_samples': n_samples,
            'n_trials': n_trials,
            'subsample': subsample
        },
        'data': {}
    }
    
    for distribution in distributions:
        logger.info(f"\nProcessing distribution: {distribution}")
        results['data'][distribution] = {}
        
        for d in tqdm(dimensions, desc=f"{distribution}"):
            # Run trials in parallel
            trial_results = Parallel(n_jobs=n_jobs)(
                delayed(run_single_trial)(distribution, n_samples, d, trial_idx, subsample)
                for trial_idx in range(n_trials)
            )
            
            # Aggregate results
            mean_distances = [r['mean_distance'] for r in trial_results]
            relative_contrasts = [r['relative_contrast'] for r in trial_results]
            nn_ratios = [r['nn_ratio'] for r in trial_results]
            cosine_means = [r['cosine_similarity_mean'] for r in trial_results]
            cosine_stds = [r['cosine_similarity_std'] for r in trial_results]
            
            dim_results = {
                'mean_distance': {
                    'mean': np.mean(mean_distances),
                    'std': np.std(mean_distances),
                    'values': mean_distances
                },
                'relative_contrast': {
                    'mean': np.mean(relative_contrasts),
                    'std': np.std(relative_contrasts),
                    'values': relative_contrasts
                },
                'nn_ratio': {
                    'mean': np.mean(nn_ratios),
                    'std': np.std(nn_ratios),
                    'values': nn_ratios
                },
                'cosine_similarity_mean': {
                    'mean': np.mean(cosine_means),
                    'std': np.std(cosine_means),
                    'values': cosine_means
                },
                'cosine_similarity_std': {
                    'mean': np.mean(cosine_stds),
                    'std': np.std(cosine_stds),
                    'values': cosine_stds
                }
            }
            
            # Collect distributions for selected dimensions
            if d in [2, 50, 500]:
                all_distances = []
                all_cosines = []
                for r in trial_results:
                    if 'distance_distribution' in r:
                        all_distances.extend(r['distance_distribution'])
                    if 'cosine_distribution' in r:
                        all_cosines.extend(r['cosine_distribution'])
                
                dim_results['distance_distribution'] = np.array(all_distances)
                dim_results['cosine_distribution'] = np.array(all_cosines)
            
            results['data'][distribution][d] = dim_results
    
    # Save results
    output_path = os.path.join(CONFIG.RESULTS_RAW, 'distance_geometry_results.pkl')
    save_results(results, output_path, format='pickle')
    
    logger.info(f"\nExperiment completed. Results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    set_seed()
    results = run_distance_geometry_experiment()
