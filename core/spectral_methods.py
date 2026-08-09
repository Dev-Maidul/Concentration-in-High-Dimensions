"""
Spectral analysis and random matrix theory utilities.
"""
import numpy as np
from typing import Tuple, Dict, Optional
from scipy import stats


def compute_sample_covariance(X: np.ndarray) -> np.ndarray:
    """
    Compute sample covariance matrix.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n, d)
    
    Returns
    -------
    np.ndarray
        Sample covariance matrix of shape (d, d)
    """
    n, d = X.shape
    X_centered = X - np.mean(X, axis=0)
    return (X_centered.T @ X_centered) / n


def compute_eigenvalues(X: np.ndarray) -> np.ndarray:
    """
    Compute eigenvalues of sample covariance matrix.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n, d)
    
    Returns
    -------
    np.ndarray
        Eigenvalues in descending order
    """
    S = compute_sample_covariance(X)
    eigenvalues = np.linalg.eigvalsh(S)
    return np.sort(eigenvalues)[::-1]  # Sort descending


def marchenko_pastur_density(x: np.ndarray, gamma: float, sigma: float = 1.0) -> np.ndarray:
    """
    Marchenko-Pastur density function.
    
    Parameters
    ----------
    x : np.ndarray
        Points at which to evaluate density
    gamma : float
        Aspect ratio d/n
    sigma : float
        Noise variance (default 1.0)
    
    Returns
    -------
    np.ndarray
        Density values
    """
    lambda_minus = sigma**2 * (1 - np.sqrt(gamma))**2
    lambda_plus = sigma**2 * (1 + np.sqrt(gamma))**2
    
    density = np.zeros_like(x)
    
    # MP density is supported on [lambda_minus, lambda_plus]
    mask = (x >= lambda_minus) & (x <= lambda_plus)
    
    if np.any(mask):
        density[mask] = (1 / (2 * np.pi * sigma**2 * gamma * x[mask])) * \
                       np.sqrt((lambda_plus - x[mask]) * (x[mask] - lambda_minus))
    
    return density


def compute_wasserstein_distance(empirical_eigs: np.ndarray, theoretical_density: np.ndarray,
                                 x_grid: np.ndarray) -> float:
    """
    Compute Wasserstein distance between empirical and theoretical distributions.
    
    Parameters
    ----------
    empirical_eigs : np.ndarray
        Empirical eigenvalues
    theoretical_density : np.ndarray
        Theoretical density values on grid
    x_grid : np.ndarray
        Grid points for theoretical density
    
    Returns
    -------
    float
        Wasserstein distance (1-Wasserstein metric)
    """
    from scipy.stats import wasserstein_distance
    
    # Normalize theoretical density
    dx = x_grid[1] - x_grid[0] if len(x_grid) > 1 else 1.0
    theoretical_weights = theoretical_density * dx
    theoretical_weights = theoretical_weights / np.sum(theoretical_weights)
    
    return wasserstein_distance(empirical_eigs, x_grid, 
                               v_weights=theoretical_weights)


def compute_spectral_statistics(X: np.ndarray, gamma: Optional[float] = None) -> Dict[str, any]:
    """
    Compute comprehensive spectral statistics.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n, d)
    gamma : float, optional
        Aspect ratio (if None, computed from X)
    
    Returns
    -------
    dict
        Dictionary containing spectral statistics
    """
    n, d = X.shape
    
    if gamma is None:
        gamma = d / n
    
    # Compute eigenvalues
    eigenvalues = compute_eigenvalues(X)
    
    # Theoretical MP bounds
    lambda_minus = (1 - np.sqrt(gamma))**2
    lambda_plus = (1 + np.sqrt(gamma))**2
    
    # Create grid for MP density
    x_min = max(0, lambda_minus - 0.5)
    x_max = lambda_plus + 0.5
    x_grid = np.linspace(x_min, x_max, 1000)
    mp_density = marchenko_pastur_density(x_grid, gamma)
    
    # Wasserstein distance
    w_dist = compute_wasserstein_distance(eigenvalues, mp_density, x_grid)
    
    return {
        'eigenvalues': eigenvalues,
        'gamma': gamma,
        'lambda_minus': lambda_minus,
        'lambda_plus': lambda_plus,
        'x_grid': x_grid,
        'mp_density': mp_density,
        'wasserstein_distance': w_dist,
        'largest_eigenvalue': eigenvalues[0],
        'smallest_eigenvalue': eigenvalues[-1],
        'condition_number': eigenvalues[0] / eigenvalues[-1] if eigenvalues[-1] > 0 else np.inf
    }


def compute_pca_stability(X: np.ndarray, noise_level: float, 
                         n_components: int = 10, seed: Optional[int] = None) -> Dict[str, float]:
    """
    Compute PCA stability under noise perturbation.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n, d)
    noise_level : float
        Standard deviation of Gaussian noise to add
    n_components : int
        Number of principal components to analyze
    seed : int, optional
        Random seed
    
    Returns
    -------
    dict
        Dictionary containing stability metrics
    """
    if seed is not None:
        np.random.seed(seed)
    
    n, d = X.shape
    n_components = min(n_components, d, n)
    
    # Original PCA
    S_original = compute_sample_covariance(X)
    eigvals_orig, eigvecs_orig = np.linalg.eigh(S_original)
    
    # Sort descending
    idx = np.argsort(eigvals_orig)[::-1]
    eigvals_orig = eigvals_orig[idx]
    eigvecs_orig = eigvecs_orig[:, idx]
    
    # Add noise
    X_noisy = X + np.random.randn(n, d) * noise_level
    
    # Noisy PCA
    S_noisy = compute_sample_covariance(X_noisy)
    eigvals_noisy, eigvecs_noisy = np.linalg.eigh(S_noisy)
    
    # Sort descending
    idx = np.argsort(eigvals_noisy)[::-1]
    eigvals_noisy = eigvals_noisy[idx]
    eigvecs_noisy = eigvecs_noisy[:, idx]
    
    # Compute overlaps
    overlaps = []
    for i in range(n_components):
        overlap = np.abs(eigvecs_orig[:, i] @ eigvecs_noisy[:, i])
        overlaps.append(overlap)
    
    # Explained variance change
    total_var_orig = np.sum(eigvals_orig[:n_components])
    total_var_noisy = np.sum(eigvals_noisy[:n_components])
    var_change = np.abs(total_var_noisy - total_var_orig) / total_var_orig if total_var_orig > 0 else 0
    
    return {
        'mean_overlap': np.mean(overlaps),
        'std_overlap': np.std(overlaps),
        'min_overlap': np.min(overlaps),
        'overlaps': np.array(overlaps),
        'explained_variance_change': var_change,
        'eigenvalue_change': np.mean(np.abs(eigvals_orig[:n_components] - eigvals_noisy[:n_components]))
    }
