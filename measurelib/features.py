"""Structure-extraction / ML support: weighted moments, covariance, PCA, and
the hyperplane-residual diagnostic used in KNOWLEDGE.md SS4 to track how
close a measure's support is to concentrating on the conjectured central
hyperplane H = {sum_k x_k = 3(n-1)}. Library-ifies the ad hoc PCA scripts
(scripts/analyze_hyperplane.py etc.) so future ML/structure-search code has
one tested implementation to call.
"""

import numpy as np

from measurelib.measure import N_COORDS


def mean(mu):
    """Weighted mean of the support in R^6."""
    return mu.weights @ mu.points


def covariance(mu):
    """Weighted covariance matrix (6, 6) of the support in R^6."""
    centered = mu.points - mean(mu)
    return (mu.weights[:, None] * centered).T @ centered


def pca(mu):
    """Eigendecomposition of the weighted covariance, sorted ascending by
    eigenvalue. Returns (eigenvalues (6,), eigenvectors (6, 6) with
    eigenvectors[:, i] the vector for eigenvalues[i]). KNOWLEDGE.md SS4 notes
    that near-optimal measures have their smallest-eigenvalue eigenvector
    converge to the all-ones direction as J approaches the known bounds --
    this is the diagnostic that observation is built from."""
    cov = covariance(mu)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    return eigenvalues, eigenvectors


def hyperplane_residual(mu, normal=None, offset=None):
    """Signed residual normal . x - offset for every support point, i.e. its
    distance (up to ||normal||) from the hyperplane {normal . x = offset}.
    Defaults to the conjectured central hyperplane sum_k x_k = 3(n-1)
    (measure.tex SS3's codimension-1 concentration conjecture, discretized).
    A measure concentrated on H has residuals ~0 for its whole support."""
    if normal is None:
        normal = np.ones(N_COORDS)
    if offset is None:
        offset = 3.0 * (mu.n - 1)
    return mu.points @ normal - offset
