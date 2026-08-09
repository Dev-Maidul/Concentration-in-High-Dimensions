"""
Norm Concentration Experiment
Analyzes norm concentration phenomena in high dimensions.
"""
import numpy as np
import logging
from typing import Dict, List
from tqdm import tqdm
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config.global_config import CONFIG, set_seed
from core.random_generators import generate_data
from core.metrics import compute_norm_statistics
from core.reproducibility import ExperimentTracker, setup_logging, save_results
from joblib import Parallel, delayed


def run_single_trial(distribution: str, n: int, d: int, trial_idx: int) -> Dict:
    """
    Run single trial of norm concentration experiment.
    
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
        Norm statistics for this trial
    """
    seed = CONFIG.RANDOM_SEED + trial_idx
    X = generate_data(distribution, n, d, seed=seed)
    stats = compute_norm_statistics(X)
    
    # Also return raw norms for histogram
    norms = np.linalg.norm(X, axis=1)
    stats['norms'] = norms
    
    return stats


def run_norm_concentration_experiment(distributions: List[str] = None,
                                     dimensions: List[int] = None,
                                     n_samples: int = None,
                                     n_trials: int = None,
                                     n_jobs: int = None,
                                     force_recompute: bool = False) -> Dict:
    """
    Run complete norm concentration experiment.
    
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
    n_jobs : int, optional
        Number of parallel jobs
    force_recompute : bool
        Force recomputation even if cached results exist
    
    Returns
    -------
    dict
        Experimental results
    """
    # Use defaults from config if not provided
    if distributions is None:
        distributions = CONFIG.DISTRIBUTIONS
    if dimensions is None:
        dimensions = CONFIG.DIMENSIONS
    if n_samples is None:
        n_samples = CONFIG.FIXED_SAMPLE_SIZE
    if n_trials is None:
        n_trials = CONFIG.N_TRIALS
    if n_jobs is None:
        n_jobs = CONFIG.N_JOBS
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting Norm Concentration Experiment")
    logger.info(f"Distributions: {distributions}")
    logger.info(f"Dimensions: {dimensions}")
    logger.info(f"Samples: {n_samples}, Trials: {n_trials}")
    
    results = {
        'config': {
            'distributions': distributions,
            'dimensions': dimensions,
            'n_samples': n_samples,
            'n_trials': n_trials
        },
        'data': {}
    }
    
    for distribution in distributions:
        logger.info(f"\nProcessing distribution: {distribution}")
        results['data'][distribution] = {}
        
        for d in tqdm(dimensions, desc=f"{distribution}"):
            # Run trials in parallel
            trial_results = Parallel(n_jobs=n_jobs)(
                delayed(run_single_trial)(distribution, n_samples, d, trial_idx)
                for trial_idx in range(n_trials)
            )
            
            # Aggregate results
            mean_norms = [r['mean_norm'] for r in trial_results]
            std_norms = [r['std_norm'] for r in trial_results]
            relative_variances = [r['relative_variance'] for r in trial_results]
            shell_thicknesses = [r['shell_thickness'] for r in trial_results]
            
            # Collect all norms for selected dimensions (for histograms)
            if d in [2, 50, 500, 2000]:
                all_norms = np.concatenate([r['norms'] for r in trial_results])
            else:
                all_norms = None
            
            results['data'][distribution][d] = {
                'mean_norm': {
                    'mean': np.mean(mean_norms),
                    'std': np.std(mean_norms),
                    'values': mean_norms
                },
                'std_norm': {
                    'mean': np.mean(std_norms),
                    'std': np.std(std_norms),
                    'values': std_norms
                },
                'relative_variance': {
                    'mean': np.mean(relative_variances),
                    'std': np.std(relative_variances),
                    'values': relative_variances
                },
                'shell_thickness': {
                    'mean': np.mean(shell_thicknesses),
                    'std': np.std(shell_thicknesses),
                    'values': shell_thicknesses
                },
                'all_norms': all_norms
            }
    
    # Save results
    output_path = os.path.join(CONFIG.RESULTS_RAW, 'norm_concentration_results.pkl')
    save_results(results, output_path, format='pickle')
    
    logger.info(f"\nExperiment completed. Results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    set_seed()
    results = run_norm_concentration_experiment()
