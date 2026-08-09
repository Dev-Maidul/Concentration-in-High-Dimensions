"""
Random data generators for various distributions.
"""
import numpy as np
from typing import Tuple, Optional
from config.global_config import CONFIG


def generate_gaussian(n: int, d: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate samples from standard Gaussian distribution.
    
    Parameters
    ----------
    n : int
        Number of samples
    d : int
        Dimension
    seed : int, optional
        Random seed
    
    Returns
    -------
    np.ndarray
        Array of shape (n, d) with Gaussian samples
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(n, d)


def generate_uniform(n: int, d: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate samples from uniform distribution on [-1, 1]^d.
    
    Parameters
    ----------
    n : int
        Number of samples
    d : int
        Dimension
    seed : int, optional
        Random seed
    
    Returns
    -------
    np.ndarray
        Array of shape (n, d) with uniform samples
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(-1, 1, size=(n, d))


def generate_laplace(n: int, d: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate samples from Laplace distribution.
    
    Parameters
    ----------
    n : int
        Number of samples
    d : int
        Dimension
    seed : int, optional
        Random seed
    
    Returns
    -------
    np.ndarray
        Array of shape (n, d) with Laplace samples
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.laplace(0, 1, size=(n, d))


def generate_student_t(n: int, d: int, df: int = None, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate samples from Student-t distribution.
    
    Parameters
    ----------
    n : int
        Number of samples
    d : int
        Dimension
    df : int, optional
        Degrees of freedom (default from CONFIG)
    seed : int, optional
        Random seed
    
    Returns
    -------
    np.ndarray
        Array of shape (n, d) with Student-t samples
    """
    if df is None:
        df = CONFIG.T_DF
    if seed is not None:
        np.random.seed(seed)
    return np.random.standard_t(df, size=(n, d))


def generate_data(distribution: str, n: int, d: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate data from specified distribution.
    
    Parameters
    ----------
    distribution : str
        Distribution name: 'gaussian', 'uniform', 'laplace', 'student_t'
    n : int
        Number of samples
    d : int
        Dimension
    seed : int, optional
        Random seed
    
    Returns
    -------
    np.ndarray
        Array of shape (n, d) with samples
    
    Raises
    ------
    ValueError
        If distribution name is not recognized
    """
    generators = {
        'gaussian': generate_gaussian,
        'uniform': generate_uniform,
        'laplace': generate_laplace,
        'student_t': generate_student_t
    }
    
    if distribution not in generators:
        raise ValueError(f"Unknown distribution: {distribution}. "
                        f"Available: {list(generators.keys())}")
    
    return generators[distribution](n, d, seed)


def generate_aspect_ratio_data(distribution: str, d: int, gamma: float, 
                               seed: Optional[int] = None) -> np.ndarray:
    """
    Generate data with specified aspect ratio gamma = d/n.
    
    Parameters
    ----------
    distribution : str
        Distribution name
    d : int
        Dimension
    gamma : float
        Aspect ratio (d/n)
    seed : int, optional
        Random seed
    
    Returns
    -------
    np.ndarray
        Array of shape (n, d) where n = d/gamma
    """
    n = int(d / gamma)
    return generate_data(distribution, n, d, seed)
