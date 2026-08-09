"""
Core metrics computation functions for high-dimensional geometry analysis.
"""
import numpy as np
from typing import Tuple, Dict, Optional
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors


def compute_norms(X: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean norms of vectors.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n, d)
    
    Returns
    -------
    np.ndarray
        Array of norms, shape (n,)
    """
    return np.linalg.norm(X, axis=1)


def compute_norm_statistics(X: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive norm statistics.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n, d)
    
    Returns
    -------
    dict
        Dictionary containing:
        - mean_norm: Mean of norms
        - std_norm: Standard deviation of norms
        - relative_variance: std / mean
        - shell_thickness: (max - min) / mean
    """
    norms = compute_norms(X)
    
    mean_norm = np.mean(norms)
    std_norm = np.std(norms)
    min_norm = np.min(norms)
    max_norm = np.max(norms)
    
    return {
        'mean_norm': mean_norm,
        'std_norm': std_norm,
        'relative_variance': std_norm / mean_norm if mean_norm > 0 else 0,
        'shell_thickness': (max_norm - min_norm) / mean_norm if mean_norm > 0 else 0,
        'min_norm': min_norm,
        'max_norm': max_norm
    }


def compute_pairwise_distances(X: np.ndarray, subsample: Optional[int] = None) -> np.ndarray:
    """
    Compute pairwise distances with optional subsampling for efficiency.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n, d)
    subsample : int, optional
        If provided, randomly subsample this many points
    
    Returns
    -------
    np.ndarray
        Pairwise distances
    """
    if subsample is not None and subsample < X.shape[0]:
        idx = np.random.choice(X.shape[0], subsample, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X
    
    return pdist(X_sub, metric='euclidean')


def compute_distance_statistics(X: np.ndarray, subsample: Optional[int] = None) -> Dict[str, float]:
    """
    Compute comprehensive distance statistics.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n, d)
    subsample : int, optional
        Number of points to subsample for efficiency
    
    Returns
    -------
    dict
        Dictionary containing distance statistics
    """
    distances = compute_pairwise_distances(X, subsample)
    
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    
    return {
        'mean_distance': mean_dist,
        'std_distance': std_dist,
        'relative_variance': std_dist / mean_dist if mean_dist > 0 else 0,
        'relative_contrast': (max_dist - min_dist) / min_dist if min_dist > 0 else 0,
        'min_distance': min_dist,
        'max_distance': max_dist
    }


def compute_nearest_neighbor_statistics(X: np.ndarray, k: int = 1) -> Dict[str, np.ndarray]:
    """
    Compute nearest and farthest neighbor statistics.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n, d)
    k : int
        Number of nearest neighbors to consider
    
    Returns
    -------
    dict
        Dictionary containing:
        - nn_distances: Nearest neighbor distances
        - fn_distances: Farthest neighbor distances
        - nn_ratio: NN distance / FN distance
    """
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    # Exclude self (first neighbor)
    nn_distances = distances[:, 1]
    
    # Farthest neighbor: approximate using sample statistics
    all_distances = compute_pairwise_distances(X[:min(1000, X.shape[0])])
    fn_distance = np.max(all_distances)
    
    return {
        'nn_distances': nn_distances,
        'mean_nn_distance': np.mean(nn_distances),
        'std_nn_distance': np.std(nn_distances),
        'fn_distance_approx': fn_distance
    }


def compute_cosine_similarity(X: np.ndarray, subsample: Optional[int] = None) -> np.ndarray:
    """
    Compute pairwise cosine similarities.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n, d)
    subsample : int, optional
        Number of points to subsample
    
    Returns
    -------
    np.ndarray
        Cosine similarities
    """
    if subsample is not None and subsample < X.shape[0]:
        idx = np.random.choice(X.shape[0], subsample, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X
    
    # Normalize rows
    X_norm = X_sub / (np.linalg.norm(X_sub, axis=1, keepdims=True) + 1e-10)
    
    # Compute cosine similarity
    similarity_matrix = X_norm @ X_norm.T
    
    # Extract upper triangle (excluding diagonal)
    return similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]


def compute_hubness_statistics(X: np.ndarray, k: int = 10) -> Dict[str, float]:
    """
    Compute hubness statistics for kNN graph.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n, d)
    k : int
        Number of nearest neighbors
    
    Returns
    -------
    dict
        Dictionary containing hubness metrics
    """
    n = X.shape[0]
    
    # Build kNN graph
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    # Count how many times each point appears as a neighbor
    neighbor_counts = np.zeros(n)
    for i in range(n):
        # Exclude self (first neighbor)
        neighbors = indices[i, 1:]
        neighbor_counts[neighbors] += 1
    
    # Compute statistics
    from scipy.stats import skew
    
    skewness = skew(neighbor_counts)
    
    # Gini coefficient
    sorted_counts = np.sort(neighbor_counts)
    n_samples = len(sorted_counts)
    index = np.arange(1, n_samples + 1)
    gini = (2 * np.sum(index * sorted_counts)) / (n_samples * np.sum(sorted_counts)) - (n_samples + 1) / n_samples
    
    return {
        'mean_neighbor_count': np.mean(neighbor_counts),
        'std_neighbor_count': np.std(neighbor_counts),
        'skewness': skewness,
        'gini_coefficient': gini,
        'max_neighbor_count': np.max(neighbor_counts),
        'neighbor_counts': neighbor_counts
    }
