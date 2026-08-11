"""Restriction and integration of a measure over codimension-1 hyperplanes
and other subsets of [0, n)^6 -- the "integrals over interesting subsets"
tool. Building blocks toward evaluating J-related quantities on the central
hyperplane H (measure.tex SS3) or other candidate structures, without having
to special-case each one as a new script.
"""

import numpy as np

from measurelib.measure import Measure
from measurelib.features import hyperplane_residual


def restrict_to_hyperplane(mu, normal=None, offset=None, tol=1e-9):
    """The sub-measure of mu supported on points within `tol` of the
    hyperplane {normal . x = offset} (defaults to the conjectured central
    hyperplane, see features.hyperplane_residual). Weights are not
    renormalized -- callers that need a probability measure should divide by
    the returned mass (mu.weights.sum())."""
    residual = hyperplane_residual(mu, normal=normal, offset=offset)
    keep = np.abs(residual) <= tol
    return Measure(points=mu.points[keep], weights=mu.weights[keep], n=mu.n)


def slice_integral(mu, coordinate, value):
    """Mass of mu on the axis-aligned codim-1 slice {x_coordinate == value}."""
    return float(mu.weights[mu.points[:, coordinate] == value].sum())


def integrate_subset(mu, predicate):
    """Mass of mu on {x : predicate(x) is True}, for an arbitrary boolean
    predicate over grid points (points -> bool array, vectorized over an
    (M, 6) array of points). The general "integral over a subset" tool --
    slice_integral and restrict_to_hyperplane are the two special cases
    common enough to warrant their own name."""
    keep = predicate(mu.points)
    return float(mu.weights[keep].sum())
