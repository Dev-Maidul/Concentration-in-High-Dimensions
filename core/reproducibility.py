"""
Reproducibility utilities for managing experiments, results, and logging.
"""
import os
import json
import pickle
import logging
from datetime import datetime
from typing import Any, Dict
import numpy as np


def setup_logging(log_file: str = None, level: str = 'INFO'):
    """
    Setup logging configuration.
    
    Parameters
    ----------
    log_file : str, optional
        Path to log file
    level : str
        Logging level
    """
    log_level = getattr(logging, level.upper())
    
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def save_results(data: Any, filepath: str, format: str = 'pickle'):
    """
    Save experimental results to disk.
    
    Parameters
    ----------
    data : any
        Data to save
    filepath : str
        Path to save file
    format : str
        Format: 'pickle', 'json', 'npy'
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if format == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    elif format == 'json':
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=convert_to_serializable)
    elif format == 'npy':
        np.save(filepath, data)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    logging.info(f"Saved results to: {filepath}")


def load_results(filepath: str, format: str = 'pickle') -> Any:
    """
    Load experimental results from disk.
    
    Parameters
    ----------
    filepath : str
        Path to load file
    format : str
        Format: 'pickle', 'json', 'npy'
    
    Returns
    -------
    any
        Loaded data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    if format == 'pickle':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    elif format == 'json':
        with open(filepath, 'r') as f:
            data = json.load(f)
    elif format == 'npy':
        data = np.load(filepath, allow_pickle=True)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    logging.info(f"Loaded results from: {filepath}")
    return data


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return str(obj)


def create_experiment_metadata(experiment_name: str, config: Dict) -> Dict:
    """
    Create metadata dictionary for experiment.
    
    Parameters
    ----------
    experiment_name : str
        Name of experiment
    config : dict
        Configuration parameters
    
    Returns
    -------
    dict
        Metadata dictionary
    """
    metadata = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'config': config
    }
    return metadata


def cache_or_compute(cache_path: str, compute_fn, force_recompute: bool = False, 
                    format: str = 'pickle'):
    """
    Cache computation results or load from cache.
    
    Parameters
    ----------
    cache_path : str
        Path to cache file
    compute_fn : callable
        Function to compute results
    force_recompute : bool
        If True, ignore cache and recompute
    format : str
        Format for caching
    
    Returns
    -------
    any
        Computation results
    """
    if not force_recompute and os.path.exists(cache_path):
        logging.info(f"Loading cached results from: {cache_path}")
        return load_results(cache_path, format=format)
    else:
        logging.info(f"Computing results...")
        results = compute_fn()
        save_results(results, cache_path, format=format)
        return results


class ExperimentTracker:
    """Track experiment progress and results."""
    
    def __init__(self, experiment_name: str, results_dir: str = 'results/raw'):
        """
        Initialize experiment tracker.
        
        Parameters
        ----------
        experiment_name : str
            Name of experiment
        results_dir : str
            Directory for results
        """
        self.experiment_name = experiment_name
        self.results_dir = results_dir
        self.start_time = datetime.now()
        self.results = {}
        
        os.makedirs(results_dir, exist_ok=True)
        
        logging.info(f"Started experiment: {experiment_name}")
    
    def log_result(self, key: str, value: Any):
        """Log a result."""
        self.results[key] = value
        logging.info(f"{key}: {value}")
    
    def save(self, filename: str = None):
        """Save all results."""
        if filename is None:
            timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.experiment_name}_{timestamp}.pkl"
        
        filepath = os.path.join(self.results_dir, filename)
        
        metadata = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_seconds': (datetime.now() - self.start_time).total_seconds()
        }
        
        full_results = {
            'metadata': metadata,
            'results': self.results
        }
        
        save_results(full_results, filepath, format='pickle')
        
        return filepath
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.save()
            logging.info(f"Experiment completed: {self.experiment_name}")
        else:
            logging.error(f"Experiment failed: {self.experiment_name}")
