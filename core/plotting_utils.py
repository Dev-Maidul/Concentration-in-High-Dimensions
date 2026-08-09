"""
Plotting utilities for consistent, publication-quality figures.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import List, Optional, Dict, Tuple
import os


# Set consistent plotting style
def setup_plotting_style():
    """Configure matplotlib for publication-quality plots."""
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 10
    rcParams['axes.labelsize'] = 11
    rcParams['axes.titlesize'] = 12
    rcParams['xtick.labelsize'] = 9
    rcParams['ytick.labelsize'] = 9
    rcParams['legend.fontsize'] = 9
    rcParams['figure.titlesize'] = 13
    rcParams['lines.linewidth'] = 1.5
    rcParams['lines.markersize'] = 6
    rcParams['figure.dpi'] = 100
    rcParams['savefig.dpi'] = 300
    rcParams['savefig.bbox'] = 'tight'
    rcParams['axes.grid'] = True
    rcParams['grid.alpha'] = 0.3
    rcParams['grid.linestyle'] = '--'


setup_plotting_style()


def save_figure(fig, filepath: str, formats: List[str] = ['png', 'pdf'], dpi: int = 300):
    """
    Save figure in multiple formats.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    filepath : str
        Base filepath (without extension)
    formats : list
        List of file formats
    dpi : int
        Resolution for raster formats
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    for fmt in formats:
        full_path = f"{filepath}.{fmt}"
        fig.savefig(full_path, format=fmt, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {full_path}")


def plot_histogram_evolution(data_dict: Dict[int, np.ndarray], 
                            dimensions: List[int],
                            xlabel: str = "Value",
                            title: str = "Distribution Evolution",
                            filepath: str = None) -> plt.Figure:
    """
    Plot histogram evolution across dimensions.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary mapping dimension to data array
    dimensions : list
        Dimensions to plot
    xlabel : str
        X-axis label
    title : str
        Plot title
    filepath : str, optional
        Path to save figure
    
    Returns
    -------
    matplotlib.figure.Figure
    """
    n_dims = len(dimensions)
    fig, axes = plt.subplots(1, n_dims, figsize=(4*n_dims, 3))
    
    if n_dims == 1:
        axes = [axes]
    
    for ax, d in zip(axes, dimensions):
        if d in data_dict:
            data = data_dict[d]
            ax.hist(data, bins=50, density=True, alpha=0.7, edgecolor='black')
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Density")
            ax.set_title(f"d = {d}")
            ax.grid(True, alpha=0.3)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if filepath:
        save_figure(fig, filepath)
    
    return fig


def plot_scaling_curve(dimensions: np.ndarray, 
                       values: np.ndarray,
                       errors: Optional[np.ndarray] = None,
                       xlabel: str = "Dimension",
                       ylabel: str = "Value",
                       title: str = "",
                       logx: bool = False,
                       logy: bool = False,
                       filepath: str = None,
                       fit_line: Optional[Tuple[float, float]] = None,
                       label: str = None) -> plt.Figure:
    """
    Plot scaling curve with optional error bars and fit line.
    
    Parameters
    ----------
    dimensions : np.ndarray
        Dimension values
    values : np.ndarray
        Metric values
    errors : np.ndarray, optional
        Error bars (standard deviation)
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    title : str
        Plot title
    logx : bool
        Use log scale for x-axis
    logy : bool
        Use log scale for y-axis
    filepath : str, optional
        Path to save figure
    fit_line : tuple, optional
        (coefficient, exponent) for power law fit line
    label : str, optional
        Label for data series
    
    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if errors is not None:
        ax.errorbar(dimensions, values, yerr=errors, fmt='o-', 
                   capsize=5, capthick=2, label=label)
    else:
        ax.plot(dimensions, values, 'o-', label=label)
    
    # Add fit line if provided
    if fit_line is not None:
        a, b = fit_line
        x_fit = np.logspace(np.log10(dimensions.min()), np.log10(dimensions.max()), 100)
        y_fit = a * np.power(x_fit, b)
        ax.plot(x_fit, y_fit, '--', color='red', alpha=0.7, 
               label=f'Fit: ${a:.2e} \\cdot d^{{{b:.2f}}}$')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    
    if label or fit_line:
        ax.legend()
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if filepath:
        save_figure(fig, filepath)
    
    return fig


def plot_multi_distribution_comparison(dimensions: np.ndarray,
                                      data_dict: Dict[str, np.ndarray],
                                      errors_dict: Optional[Dict[str, np.ndarray]] = None,
                                      xlabel: str = "Dimension",
                                      ylabel: str = "Value",
                                      title: str = "",
                                      logx: bool = False,
                                      logy: bool = False,
                                      filepath: str = None) -> plt.Figure:
    """
    Plot comparison of multiple distributions.
    
    Parameters
    ----------
    dimensions : np.ndarray
        Dimension values
    data_dict : dict
        Dictionary mapping distribution name to values
    errors_dict : dict, optional
        Dictionary mapping distribution name to errors
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    title : str
        Plot title
    logx : bool
        Use log scale for x-axis
    logy : bool
        Use log scale for y-axis
    filepath : str, optional
        Path to save figure
    
    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for dist_name, values in data_dict.items():
        if errors_dict and dist_name in errors_dict:
            errors = errors_dict[dist_name]
            ax.errorbar(dimensions, values, yerr=errors, fmt='o-', 
                       label=dist_name, capsize=3)
        else:
            ax.plot(dimensions, values, 'o-', label=dist_name)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if filepath:
        save_figure(fig, filepath)
    
    return fig


def plot_phase_transition(x_values: np.ndarray,
                         y_values: np.ndarray,
                         threshold: float,
                         xlabel: str = "Parameter",
                         ylabel: str = "Metric",
                         title: str = "Phase Transition",
                         filepath: str = None) -> plt.Figure:
    """
    Plot phase transition with threshold line.
    
    Parameters
    ----------
    x_values : np.ndarray
        X-axis values
    y_values : np.ndarray
        Y-axis values
    threshold : float
        Threshold value for phase transition
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    title : str
        Plot title
    filepath : str, optional
        Path to save figure
    
    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(x_values, y_values, 'o-', linewidth=2)
    ax.axhline(y=threshold, color='red', linestyle='--', 
              label=f'Threshold = {threshold}')
    
    # Find approximate transition point
    if len(y_values) > 0:
        transition_idx = np.argmin(np.abs(y_values - threshold))
        if transition_idx > 0 and transition_idx < len(x_values):
            ax.axvline(x=x_values[transition_idx], color='green', 
                      linestyle=':', alpha=0.5,
                      label=f'Transition ≈ {x_values[transition_idx]:.1f}')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if filepath:
        save_figure(fig, filepath)
    
    return fig


def plot_2d_heatmap(data: np.ndarray,
                   x_labels: List,
                   y_labels: List,
                   xlabel: str = "X",
                   ylabel: str = "Y",
                   title: str = "",
                   cmap: str = 'viridis',
                   filepath: str = None) -> plt.Figure:
    """
    Plot 2D heatmap.
    
    Parameters
    ----------
    data : np.ndarray
        2D data array
    x_labels : list
        X-axis tick labels
    y_labels : list
        Y-axis tick labels
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    title : str
        Plot title
    cmap : str
        Colormap name
    filepath : str, optional
        Path to save figure
    
    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(data, cmap=cmap, aspect='auto')
    
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    if filepath:
        save_figure(fig, filepath)
    
    return fig
