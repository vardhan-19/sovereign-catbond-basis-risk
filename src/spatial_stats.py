"""
Spatial econometric diagnostics: Moran's I and Conley (1999) Spatial HAC Covariance.
"""

from typing import Dict, Tuple
from esda.moran import Moran
from libpysal.weights import KNN
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def calculate_morans_i(
    residuals: np.ndarray,
    coords: np.ndarray,
    k_neighbors: int = 5,
    permutations: int = 999
) -> Dict[str, float]:
    """
    Evaluate Global Moran's I statistic on regression residuals.

    Parameters
    ----------
    residuals : ndarray
        1D array of regression residuals (e_i).
    coords : ndarray
        N x 2 array of metric coordinates (centroid_x, centroid_y).
    k_neighbors : int
        Number of nearest neighbors (default: 5).
    permutations : int
        Monte Carlo spatial permutations (default: 999).

    Returns
    -------
    dict
        Moran's I, Expected I, z-score, and permutation p-value.
    """
    w = KNN.from_array(coords, k=k_neighbors)
    w.transform = "r"  # Row-standardize

    moran = Moran(residuals, w, permutations=permutations)

    return {
        "morans_i": float(moran.I),
        "expected_i": float(moran.EI),
        "z_score": float(moran.z_sim),
        "p_value": float(moran.p_sim)
    }


def conley_spatial_hac_ols(
    X: np.ndarray,
    residuals: np.ndarray,
    coords: np.ndarray,
    distance_cutoff_km: float = 50.0,
    kernel_type: str = "bartlett",
    finite_sample: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute non-parametric Conley (1999) Spatial HAC standard errors for OLS.

    Parameters
    ----------
    X : ndarray
        N x K design matrix (including intercept column).
    residuals : ndarray
        N-length array of OLS residuals (e_i = y_i - X_i * beta).
    coords : ndarray
        N x 2 array of metric coordinates in meters.
    distance_cutoff_km : float
        Spatial cutoff bandwidth in kilometers.
    kernel_type : str
        'bartlett' (triangular, positive semi-definite) or 'uniform'.
    finite_sample : bool
        If True, applies n / (n - k) degrees-of-freedom scaling to harmonize
        with Huber-White HC1 and unadjusted OLS estimators.

    Returns
    -------
    Tuple[ndarray, ndarray]
        (se_conley, V_conley) where se_conley is 1D array of standard errors.
    """
    n, k = X.shape

    # Pairwise metric Euclidean distances converted to kilometers
    dist_matrix_km = cdist(coords, coords, metric="euclidean") / 1000.0

    # Weighting kernel
    if kernel_type.lower() == "bartlett":
        kernel = np.maximum(0.0, 1.0 - (dist_matrix_km / distance_cutoff_km))
    else:  # Uniform
        kernel = (dist_matrix_km <= distance_cutoff_km).astype(float)

    # Meat of the sandwich estimator: Sum_i Sum_j K(d_ij) * e_i * e_j * x_i * x_j'
    e_X = residuals.reshape(-1, 1) * X  # N x K matrix
    meat = np.zeros((k, k))
    for i in range(n):
        weights_i = kernel[i, :].reshape(-1, 1)  # N x 1
        weighted_e_X = e_X * weights_i           # N x K
        meat += np.outer(e_X[i], weighted_e_X.sum(axis=0))

    # Finite-sample degrees of freedom correction: n / (n - k)
    if finite_sample and n > k:
        meat = meat * (n / (n - k))

    # Bread: (X'X)^(-1)
    bread = np.linalg.inv(X.T @ X)

    # Sandwich covariance: V = bread @ meat @ bread
    V_conley = bread @ meat @ bread

    # Spectral Positive Semi-Definite (PSD) projection (Kelejian & Prucha 2007)
    eigvals, eigvecs = np.linalg.eigh(V_conley)
    if np.any(eigvals < 0):
        eigvals_psd = np.maximum(eigvals, 0.0)
        V_conley = eigvecs @ np.diag(eigvals_psd) @ eigvecs.T

    se_conley = np.sqrt(np.maximum(0.0, np.diag(V_conley)))

    return se_conley, V_conley


def conley_spatial_hac_glm(
    hessian_inv: np.ndarray,
    quasi_scores: np.ndarray,
    coords: np.ndarray,
    distance_cutoff_km: float = 50.0,
    kernel_type: str = "bartlett",
    finite_sample: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Conley (1999) Spatial HAC standard errors for Fractional Logit (QMLE GLM).

    Parameters
    ----------
    hessian_inv : ndarray
        K x K inverse Hessian matrix from GLM fit (the 'bread').
    quasi_scores : ndarray
        N x K empirical quasi-score contributions (s_i = (y_i - mu_i) / (mu_i*(1-mu_i)) * d_mu).
    coords : ndarray
        N x 2 array of metric coordinates in meters.
    distance_cutoff_km : float
        Spatial cutoff bandwidth in kilometers.
    kernel_type : str
        'bartlett' or 'uniform'.
    finite_sample : bool
        If True, applies n / (n - k) degrees-of-freedom scaling.

    Returns
    -------
    Tuple[ndarray, ndarray]
        (se_conley_glm, V_conley_glm)
    """
    n, k = quasi_scores.shape
    dist_matrix_km = cdist(coords, coords, metric="euclidean") / 1000.0

    if kernel_type.lower() == "bartlett":
        kernel = np.maximum(0.0, 1.0 - (dist_matrix_km / distance_cutoff_km))
    else:
        kernel = (dist_matrix_km <= distance_cutoff_km).astype(float)

    meat = np.zeros((k, k))
    for i in range(n):
        weights_i = kernel[i, :].reshape(-1, 1)
        weighted_scores = quasi_scores * weights_i
        meat += np.outer(quasi_scores[i], weighted_scores.sum(axis=0))

    if finite_sample and n > k:
        meat = meat * (n / (n - k))

    V_conley_glm = hessian_inv @ meat @ hessian_inv

    # Spectral Positive Semi-Definite (PSD) projection
    eigvals, eigvecs = np.linalg.eigh(V_conley_glm)
    if np.any(eigvals < 0):
        eigvals_psd = np.maximum(eigvals, 0.0)
        V_conley_glm = eigvecs @ np.diag(eigvals_psd) @ eigvecs.T

    se_conley_glm = np.sqrt(np.maximum(0.0, np.diag(V_conley_glm)))

    return se_conley_glm, V_conley_glm
