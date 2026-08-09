"""
Johnson-Lindenstrauss projection methods.
"""
import numpy as np
from typing import Tuple, Optional
import time


def gaussian_projection(X: np.ndarray, k: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gaussian random projection.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n, d)
    k : int
        Target projection dimension
    seed : int, optional
        Random seed
    
    Returns
    -------
    tuple
        (projected_data, projection_matrix)
    """
    n, d = X.shape
    
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random Gaussian projection matrix
    R = np.random.randn(d, k) / np.sqrt(k)
    
    # Project data
    X_proj = X @ R
    
    return X_proj, R


def sparse_projection(X: np.ndarray, k: int, sparsity: float = 0.9, 
                     seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sparse random projection (Achlioptas).
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n, d)
    k : int
        Target projection dimension
    sparsity : float
        Fraction of zero entries (default 0.9)
    seed : int, optional
        Random seed
    
    Returns
    -------
    tuple
        (projected_data, projection_matrix)
    """
    n, d = X.shape
    
    if seed is not None:
        np.random.seed(seed)
    
    # Create sparse projection matrix
    R = np.zeros((d, k))
    
    # Probability distribution
    s = 1.0 / (1 - sparsity)
    
    for i in range(d):
        for j in range(k):
            r = np.random.rand()
            if r < (1 - sparsity) / 2:
                R[i, j] = np.sqrt(s)
            elif r < (1 - sparsity):
                R[i, j] = -np.sqrt(s)
            # else: remains 0
    
    R = R / np.sqrt(k)
    
    # Project data
    X_proj = X @ R
    
    return X_proj, R


def structured_projection(X: np.ndarray, k: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Structured random projection (simplified Fast JL).
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n, d)
    k : int
        Target projection dimension
    seed : int, optional
        Random seed
    
    Returns
    -------
    tuple
        (projected_data, projection_matrix)
    """
    n, d = X.shape
    
    if seed is not None:
        np.random.seed(seed)
    
    # Random diagonal scaling
    D = np.random.choice([-1, 1], size=d)
    
    # Apply diagonal scaling
    X_scaled = X * D
    
    # Random subsampling (simplified structured approach)
    if k < d:
        idx = np.random.choice(d, k, replace=False)
        X_proj = X_scaled[:, idx] * np.sqrt(d / k)
        R = np.zeros((d, k))
        R[idx, np.arange(k)] = D[idx] * np.sqrt(d / k)
    else:
        X_proj = X_scaled
        R = np.diag(D)
    
    return X_proj, R


def compute_projection_distortion(X: np.ndarray, X_proj: np.ndarray, 
                                 subsample: Optional[int] = None) -> np.ndarray:
    """
    Compute distortion for projected distances.
    
    Parameters
    ----------
    X : np.ndarray
        Original data of shape (n, d)
    X_proj : np.ndarray
        Projected data of shape (n, k)
    subsample : int, optional
        Number of pairs to subsample for efficiency
    
    Returns
    -------
    np.ndarray
        Array of distortion values (projected_dist / original_dist)
    """
    from scipy.spatial.distance import pdist
    
    # Subsample if needed
    if subsample is not None and X.shape[0] > subsample:
        idx = np.random.choice(X.shape[0], subsample, replace=False)
        X_sub = X[idx]
        X_proj_sub = X_proj[idx]
    else:
        X_sub = X
        X_proj_sub = X_proj
    
    # Compute pairwise distances
    original_dists = pdist(X_sub, metric='euclidean')
    projected_dists = pdist(X_proj_sub, metric='euclidean')
    
    # Avoid division by zero
    mask = original_dists > 1e-10
    distortion = np.zeros_like(original_dists)
    distortion[mask] = projected_dists[mask] / original_dists[mask]
    
    return distortion[mask]


def compute_failure_probability(distortions: np.ndarray, epsilon: float) -> float:
    """
    Compute probability that distortion exceeds epsilon.
    
    Parameters
    ----------
    distortions : np.ndarray
        Array of distortion values
    epsilon : float
        Distortion threshold
    
    Returns
    -------
    float
        Fraction of distances with distortion > epsilon
    """
    failures = np.abs(distortions - 1.0) > epsilon
    return np.mean(failures)


def project_with_method(X: np.ndarray, k: int, method: str, 
                       seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Apply projection method and measure computation time.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix
    k : int
        Target dimension
    method : str
        Projection method: 'gaussian', 'sparse', 'structured'
    seed : int, optional
        Random seed
    
    Returns
    -------
    tuple
        (projected_data, projection_matrix, computation_time)
    """
    methods = {
        'gaussian': gaussian_projection,
        'sparse': sparse_projection,
        'structured': structured_projection
    }
    
    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Available: {list(methods.keys())}")
    
    start_time = time.time()
    X_proj, R = methods[method](X, k, seed)
    elapsed = time.time() - start_time
    
    return X_proj, R, elapsed
