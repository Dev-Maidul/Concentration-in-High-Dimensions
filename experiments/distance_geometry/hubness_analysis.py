"""
Hubness Analysis
Analyzes hubness phenomenon in high-dimensional nearest neighbor graphs.
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
from core.metrics import compute_hubness_statistics
from core.reproducibility import setup_logging, save_results
from joblib import Parallel, delayed


def run_single_hubness_trial(distribution: str, n: int, d: int, 
                            trial_idx: int, k: int = 10) -> Dict:
    """
    Run single trial of hubness analysis.
    
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
    k : int
        Number of nearest neighbors
    
    Returns
    -------
    dict
        Hubness statistics for this trial
    """
    seed = CONFIG.RANDOM_SEED + trial_idx
    X = generate_data(distribution, n, d, seed=seed)
    
    hub_stats = compute_hubness_statistics(X, k=k)
    
    # For selected dimensions, save neighbor count distribution
    if d in [2, 50, 500, 2000]:
        neighbor_counts = hub_stats.pop('neighbor_counts')
        hub_stats['neighbor_count_distribution'] = neighbor_counts
    else:
        hub_stats.pop('neighbor_counts', None)
    
    return hub_stats


def run_hubness_analysis(distributions: List[str] = None,
                        dimensions: List[int] = None,
                        n_samples: int = None,
                        n_trials: int = None,
                        k_neighbors: int = None,
                        n_jobs: int = None) -> Dict:
    """
    Run complete hubness analysis.
    
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
    k_neighbors : int, optional
        Number of nearest neighbors for kNN graph
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
    if k_neighbors is None:
        k_neighbors = CONFIG.K_NEIGHBORS
    if n_jobs is None:
        n_jobs = CONFIG.N_JOBS
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting Hubness Analysis")
    logger.info(f"Distributions: {distributions}")
    logger.info(f"Dimensions: {dimensions}")
    logger.info(f"Samples: {n_samples}, k-neighbors: {k_neighbors}")
    
    results = {
        'config': {
            'distributions': distributions,
            'dimensions': dimensions,
            'n_samples': n_samples,
            'n_trials': n_trials,
            'k_neighbors': k_neighbors
        },
        'data': {}
    }
    
    for distribution in distributions:
        logger.info(f"\nProcessing distribution: {distribution}")
        results['data'][distribution] = {}
        
        for d in tqdm(dimensions, desc=f"{distribution}"):
            # Run trials in parallel
            trial_results = Parallel(n_jobs=n_jobs)(
                delayed(run_single_hubness_trial)(distribution, n_samples, d, 
                                                 trial_idx, k_neighbors)
                for trial_idx in range(n_trials)
            )
            
            # Aggregate results
            skewnesses = [r['skewness'] for r in trial_results]
            gini_coeffs = [r['gini_coefficient'] for r in trial_results]
            mean_counts = [r['mean_neighbor_count'] for r in trial_results]
            max_counts = [r['max_neighbor_count'] for r in trial_results]
            
            dim_results = {
                'skewness': {
                    'mean': np.mean(skewnesses),
                    'std': np.std(skewnesses),
                    'values': skewnesses
                },
                'gini_coefficient': {
                    'mean': np.mean(gini_coeffs),
                    'std': np.std(gini_coeffs),
                    'values': gini_coeffs
                },
                'mean_neighbor_count': {
                    'mean': np.mean(mean_counts),
                    'std': np.std(mean_counts)
                },
                'max_neighbor_count': {
                    'mean': np.mean(max_counts),
                    'std': np.std(max_counts)
                }
            }
            
            # Collect distributions for selected dimensions
            if d in [2, 50, 500, 2000]:
                all_counts = []
                for r in trial_results:
                    if 'neighbor_count_distribution' in r:
                        all_counts.extend(r['neighbor_count_distribution'])
                
                if all_counts:
                    dim_results['neighbor_count_distribution'] = np.array(all_counts)
            
            results['data'][distribution][d] = dim_results
    
    # Save results
    output_path = os.path.join(CONFIG.RESULTS_RAW, 'hubness_analysis_results.pkl')
    save_results(results, output_path, format='pickle')
    
    logger.info(f"\nHubness analysis completed. Results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    set_seed()
    results = run_hubness_analysis()
