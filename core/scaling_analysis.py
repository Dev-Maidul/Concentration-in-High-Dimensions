"""
Scaling law analysis and power-law fitting.
"""
import numpy as np
from typing import Dict, Tuple
from scipy.optimize import curve_fit
from scipy import stats


def power_law(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Power law function: y = a * x^b
    
    Parameters
    ----------
    x : np.ndarray
        Input values
    a : float
        Scaling coefficient
    b : float
        Power law exponent
    
    Returns
    -------
    np.ndarray
        Output values
    """
    return a * np.power(x, b)


def fit_power_law(x: np.ndarray, y: np.ndarray, 
                 use_log: bool = True) -> Dict[str, float]:
    """
    Fit power law to data.
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y : np.ndarray
        Dependent variable
    use_log : bool
        If True, fit in log-log space (more stable)
    
    Returns
    -------
    dict
        Dictionary containing:
        - coefficient: a in y = a * x^b
        - exponent: b in y = a * x^b
        - r_squared: Coefficient of determination
        - std_error: Standard error of exponent
    """
    # Remove any invalid values
    valid_mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    
    if len(x_valid) < 2:
        return {
            'coefficient': np.nan,
            'exponent': np.nan,
            'r_squared': np.nan,
            'std_error': np.nan
        }
    
    if use_log:
        # Fit in log-log space: log(y) = log(a) + b * log(x)
        log_x = np.log(x_valid)
        log_y = np.log(y_valid)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
        
        coefficient = np.exp(intercept)
        exponent = slope
        r_squared = r_value ** 2
        std_error = std_err
    else:
        # Direct nonlinear fit
        try:
            popt, pcov = curve_fit(power_law, x_valid, y_valid, p0=[1.0, -0.5])
            coefficient, exponent = popt
            
            # Compute R^2
            y_pred = power_law(x_valid, *popt)
            ss_res = np.sum((y_valid - y_pred) ** 2)
            ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            std_error = np.sqrt(np.diag(pcov))[1]
        except:
            return {
                'coefficient': np.nan,
                'exponent': np.nan,
                'r_squared': np.nan,
                'std_error': np.nan
            }
    
    return {
        'coefficient': coefficient,
        'exponent': exponent,
        'r_squared': r_squared,
        'std_error': std_error
    }


def exponential_decay(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Exponential decay: y = a * exp(-b * x)
    
    Parameters
    ----------
    x : np.ndarray
        Input values
    a : float
        Initial value
    b : float
        Decay rate
    
    Returns
    -------
    np.ndarray
        Output values
    """
    return a * np.exp(-b * x)


def fit_exponential(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Fit exponential decay to data.
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y : np.ndarray
        Dependent variable
    
    Returns
    -------
    dict
        Dictionary containing fit parameters
    """
    valid_mask = (y > 0) & np.isfinite(x) & np.isfinite(y)
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    
    if len(x_valid) < 2:
        return {
            'coefficient': np.nan,
            'decay_rate': np.nan,
            'r_squared': np.nan
        }
    
    try:
        popt, pcov = curve_fit(exponential_decay, x_valid, y_valid, 
                              p0=[np.max(y_valid), 0.01])
        coefficient, decay_rate = popt
        
        # Compute R^2
        y_pred = exponential_decay(x_valid, *popt)
        ss_res = np.sum((y_valid - y_pred) ** 2)
        ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'coefficient': coefficient,
            'decay_rate': decay_rate,
            'r_squared': r_squared
        }
    except:
        return {
            'coefficient': np.nan,
            'decay_rate': np.nan,
            'r_squared': np.nan
        }


def compare_scaling_laws(dimensions: np.ndarray, 
                        metrics: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    Compare scaling laws across multiple metrics.
    
    Parameters
    ----------
    dimensions : np.ndarray
        Array of dimensions
    metrics : dict
        Dictionary mapping metric names to arrays of values
    
    Returns
    -------
    dict
        Dictionary of fit results for each metric
    """
    results = {}
    
    for metric_name, values in metrics.items():
        power_fit = fit_power_law(dimensions, values)
        exp_fit = fit_exponential(dimensions, values)
        
        # Choose best fit based on R^2
        if power_fit['r_squared'] > exp_fit['r_squared']:
            best_fit = 'power_law'
            best_params = power_fit
        else:
            best_fit = 'exponential'
            best_params = exp_fit
        
        results[metric_name] = {
            'best_fit': best_fit,
            'power_law': power_fit,
            'exponential': exp_fit,
            **best_params
        }
    
    return results
