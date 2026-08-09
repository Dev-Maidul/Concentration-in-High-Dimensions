"""
Global configuration for high-dimensional geometry experiments.
"""
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np


@dataclass
class GlobalConfig:
    """Global configuration for all experiments."""
    
    # Random seed for reproducibility
    RANDOM_SEED: int = 42
    
    # Dimensions to test
    DIMENSIONS: List[int] = None
    
    # Sample sizes
    FIXED_SAMPLE_SIZE: int = 10000
    
    # Aspect ratios for scaling experiments
    ASPECT_RATIOS: List[float] = None
    
    # Number of independent trials
    N_TRIALS: int = 50
    
    # Distributions to test
    DISTRIBUTIONS: List[str] = None
    
    # Student-t degrees of freedom
    T_DF: int = 3
    
    # Result directories
    RESULTS_RAW: str = "results/raw"
    RESULTS_PROCESSED: str = "results/processed"
    RESULTS_FIGURES: str = "results/figures"
    RESULTS_TABLES: str = "results/tables"
    
    # Plotting configuration
    FIGURE_DPI: int = 300
    FIGURE_FORMAT: List[str] = None
    
    # Multiprocessing
    N_JOBS: int = -1  # Use all available cores
    
    # Distance computation subsampling
    DISTANCE_SUBSAMPLE: int = 1000
    
    # kNN parameters
    K_NEIGHBORS: int = 10
    
    # JL projection parameters
    JL_EPSILON_VALUES: List[float] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.DIMENSIONS is None:
            self.DIMENSIONS = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
        
        if self.ASPECT_RATIOS is None:
            self.ASPECT_RATIOS = [0.1, 0.5, 1.0, 2.0]
        
        if self.DISTRIBUTIONS is None:
            self.DISTRIBUTIONS = ['gaussian', 'uniform', 'laplace', 'student_t']
        
        if self.FIGURE_FORMAT is None:
            self.FIGURE_FORMAT = ['png', 'pdf']
        
        if self.JL_EPSILON_VALUES is None:
            self.JL_EPSILON_VALUES = [0.1, 0.2, 0.3]


# Global singleton instance
CONFIG = GlobalConfig()


def get_config() -> GlobalConfig:
    """Get the global configuration instance."""
    return CONFIG


def set_seed(seed: int = None):
    """Set random seed for reproducibility."""
    if seed is None:
        seed = CONFIG.RANDOM_SEED
    np.random.seed(seed)
    return seed
