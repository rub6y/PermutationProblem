"""Sparse representation of measures in A_n (Definition in measure.tex Sec. 4):
nonnegative weights on grid cells of [n]^6 whose 6 marginals are all uniform
(1/n each). This is the single place that owns the (points, weights, n)
representation and the code that loads/validates it.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_COORDS = 6


@dataclass
class Measure:
    """A finitely supported measure on [0, n)^6.

    points: int64 array of shape (M, 6), grid cell indices in [0, n).
    weights: float64 array of shape (M,), nonnegative, summing to 1.
    n: grid resolution.
    """

    points: np.ndarray
    weights: np.ndarray
    n: int

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=np.int64)
        self.weights = np.asarray(self.weights, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != N_COORDS:
            raise ValueError(f"points must have shape (M, {N_COORDS})")
        if self.weights.shape != (self.points.shape[0],):
            raise ValueError("weights must have shape (M,) matching points")

    @property
    def support_size(self):
        return self.points.shape[0]


def load_permutations(path):
    """Parse a file in the `good_permutation.txt` format: a Python literal
    `S1 = [ (perm as tuple), ... ]` of 6 permutations of range(n). Returns the
    list of 6 tuples (this is the one place that format is parsed)."""
    text = Path(path).read_text()
    namespace = {}
    exec(compile(text, str(path), "exec"), namespace)
    permutations = next(v for k, v in namespace.items() if not k.startswith("__"))
    if len(permutations) != N_COORDS:
        raise ValueError(f"expected {N_COORDS} permutations, got {len(permutations)}")
    n = len(permutations[0])
    for perm in permutations:
        if sorted(perm) != list(range(n)):
            raise ValueError("each row must be a permutation of range(n)")
    return permutations


def measure_from_permutations(permutations):
    """Six permutations of [n] -> the uniform measure on their n grid points
    (Definition of iota_n in measure.tex, restricted to a single 6-tuple)."""
    n = len(permutations[0])
    points = np.array(permutations, dtype=np.int64).T  # shape (n, 6)
    weights = np.full(n, 1.0 / n, dtype=np.float64)
    return Measure(points=points, weights=weights, n=n)


def load_permutations_as_measure(path):
    return measure_from_permutations(load_permutations(path))


def marginal_masses(measure):
    """For each of the 6 coordinates, the total weight landing in each of the
    n grid cells (should be uniform 1/n for measure in A_n)."""
    n = measure.n
    masses = np.zeros((N_COORDS, n), dtype=np.float64)
    for coord in range(N_COORDS):
        np.add.at(masses[coord], measure.points[:, coord], measure.weights)
    return masses


def mix(measure_a, measure_b, t):
    """The measure (1 - t) * measure_a + t * measure_b, with matching support
    points merged (weights added) rather than duplicated."""
    combined = {}
    for point, weight in zip(map(tuple, measure_a.points), measure_a.weights):
        combined[point] = combined.get(point, 0.0) + (1.0 - t) * weight
    for point, weight in zip(map(tuple, measure_b.points), measure_b.weights):
        combined[point] = combined.get(point, 0.0) + t * weight
    points = np.array(list(combined.keys()), dtype=np.int64)
    weights = np.array(list(combined.values()), dtype=np.float64)
    keep = weights > 1e-15
    return Measure(points=points[keep], weights=weights[keep], n=measure_a.n)


def check_marginals(measure, tol=1e-9):
    """Raise if `measure` is not (approximately) in A_n: nonnegative weights
    summing to 1 with all 6 marginals uniform."""
    if np.any(measure.weights < -tol):
        raise ValueError("weights must be nonnegative")
    total = measure.weights.sum()
    if abs(total - 1.0) > tol:
        raise ValueError(f"weights must sum to 1, got {total}")
    target = 1.0 / measure.n
    masses = marginal_masses(measure)
    max_err = np.max(np.abs(masses - target))
    if max_err > tol:
        raise ValueError(f"marginal {max_err} away from uniform 1/n (tol={tol})")
    return max_err
