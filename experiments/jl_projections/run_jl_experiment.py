"""
Johnson-Lindenstrauss Projections Experiment
Tests random projection methods and distortion analysis.
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
from core.projection_methods import (project_with_method, 
                                    compute_projection_distortion,
                                    compute_failure_probability)
from core.reproducibility import setup_logging, save_results
from joblib import Parallel, delayed


def run_single_jl_trial(distribution: str, n: int, d: int, k: int,
                       method: str, trial_idx: int, 
                       epsilon_values: List[float],
                       subsample: int = 500) -> Dict:
    """
    Run single trial of JL projection.
    
    Parameters
    ----------
    distribution : str
        Distribution name
    n : int
        Number of samples
    d : int
        Original dimension
    k : int
        Projection dimension
    method : str
        Projection method
    trial_idx : int
        Trial index for seeding
    epsilon_values : list
        Epsilon values for failure probability
    subsample : int
        Subsample for distance computation
    
    Returns
    -------
    dict
        Projection statistics
    """
    seed = CONFIG.RANDOM_SEED + trial_idx
    X = generate_data(distribution, n, d, seed=seed)
    
    # Apply projection
    X_proj, R, comp_time = project_with_method(X, k, method, seed=seed)
    
    # Compute distortion
    distortions = compute_projection_distortion(X, X_proj, subsample=subsample)
    
    # Compute failure probabilities
    failure_probs = {}
    for epsilon in epsilon_values:
        failure_probs[epsilon] = compute_failure_probability(distortions, epsilon)
    
    results = {
        'mean_distortion': np.mean(distortions),
        'std_distortion': np.std(distortions),
        'max_distortion': np.max(distortions),
        'min_distortion': np.min(distortions),
        'computation_time': comp_time,
        'failure_probabilities': failure_probs
    }
    
    # For selected projection dimensions, save distribution
    if k in [10, 50, 100]:
        results['distortion_distribution'] = distortions[:1000]  # Limit size
    
    return results


def run_jl_projection_experiment(distributions: List[str] = None,
                                dimensions: List[int] = None,
                                n_samples: int = None,
                                n_trials: int = None,
                                epsilon_values: List[float] = None,
                                projection_methods: List[str] = None,
                                n_jobs: int = None) -> Dict:
    """
    Run complete JL projection experiment.
    
    Parameters
    ----------
    distributions : list, optional
        List of distributions to test
    dimensions : list, optional
        List of original dimensions to test
    n_samples : int, optional
        Number of samples per trial
    n_trials : int, optional
        Number of independent trials
    epsilon_values : list, optional
        Epsilon values for failure probability
    projection_methods : list, optional
        Projection methods to compare
    n_jobs : int, optional
        Number of parallel jobs
    
    Returns
    -------
    dict
        Experimental results
    """
    if distributions is None:
        distributions = ['gaussian']  # Focus on Gaussian for JL
    if dimensions is None:
        # Use subset of dimensions for JL experiments
        dimensions = [50, 100, 200, 500, 1000]
    if n_samples is None:
        n_samples = 1000  # Smaller for JL
    if n_trials is None:
        n_trials = CONFIG.N_TRIALS
    if epsilon_values is None:
        epsilon_values = CONFIG.JL_EPSILON_VALUES
    if projection_methods is None:
        projection_methods = ['gaussian', 'sparse', 'structured']
    if n_jobs is None:
        n_jobs = CONFIG.N_JOBS
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting JL Projection Experiment")
    logger.info(f"Dimensions: {dimensions}")
    logger.info(f"Projection methods: {projection_methods}")
    logger.info(f"Epsilon values: {epsilon_values}")
    
    results = {
        'config': {
            'distributions': distributions,
            'dimensions': dimensions,
            'n_samples': n_samples,
            'n_trials': n_trials,
            'epsilon_values': epsilon_values,
            'projection_methods': projection_methods
        },
        'data': {}
    }
    
    for distribution in distributions:
        results['data'][distribution] = {}
        
        for d in dimensions:
            logger.info(f"\nOriginal dimension: {d}")
            
            # Test different projection dimensions
            # Use theoretical JL bound: k >= O(log(n) / eps^2)
            projection_dims = [
                int(d * 0.1),  # 10% of original
                int(d * 0.25), # 25% of original
                int(d * 0.5),  # 50% of original
                min(int(d * 0.75), d-1)  # 75% of original
            ]
            projection_dims = [k for k in projection_dims if k >= 2]
            
            results['data'][distribution][d] = {}
            
            for method in projection_methods:
                logger.info(f"  Method: {method}")
                results['data'][distribution][d][method] = {}
                
                for k in tqdm(projection_dims, desc=f"d={d}, {method}"):
                    # Run trials in parallel
                    trial_results = Parallel(n_jobs=n_jobs)(
                        delayed(run_single_jl_trial)(
                            distribution, n_samples, d, k, method, 
                            trial_idx, epsilon_values
                        )
                        for trial_idx in range(n_trials)
                    )
                    
                    # Aggregate results
                    mean_distortions = [r['mean_distortion'] for r in trial_results]
                    max_distortions = [r['max_distortion'] for r in trial_results]
                    comp_times = [r['computation_time'] for r in trial_results]
                    
                    # Aggregate failure probabilities
                    failure_prob_agg = {}
                    for eps in epsilon_values:
                        probs = [r['failure_probabilities'][eps] for r in trial_results]
                        failure_prob_agg[eps] = {
                            'mean': np.mean(probs),
                            'std': np.std(probs),
                            'values': probs
                        }
                    
                    k_results = {
                        'mean_distortion': {
                            'mean': np.mean(mean_distortions),
                            'std': np.std(mean_distortions),
                            'values': mean_distortions
                        },
                        'max_distortion': {
                            'mean': np.mean(max_distortions),
                            'std': np.std(max_distortions)
                        },
                        'computation_time': {
                            'mean': np.mean(comp_times),
                            'std': np.std(comp_times)
                        },
                        'failure_probabilities': failure_prob_agg
                    }
                    
                    # Collect distortion distributions
                    if k in [10, 50, 100]:
                        all_distortions = []
                        for r in trial_results:
                            if 'distortion_distribution' in r:
                                all_distortions.extend(r['distortion_distribution'])
                        
                        if all_distortions:
                            k_results['distortion_distribution'] = np.array(all_distortions)
                    
                    results['data'][distribution][d][method][k] = k_results
    
    # Save results
    output_path = os.path.join(CONFIG.RESULTS_RAW, 'jl_projection_results.pkl')
    save_results(results, output_path, format='pickle')
    
    logger.info(f"\nJL projection experiment completed. Results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    set_seed()
    results = run_jl_projection_experiment()
