"""The objective J_n (measure.tex Def. 2.2 / Sec. 4) and its gradient.

J(a) = sum_{i,j,k in support} a_i a_j a_k * K(i,j,k), K = shattered-triple
indicator (fw.shattering). As a_i are the weights of a measure, J is a cubic
polynomial. Since K is symmetric under permuting its three arguments,
dJ/da_p = 3 * sum_{q,r in support} a_q a_r K(p,q,r).

This module is the single place that turns the raw shattering predicate
(fw.shattering) into the actual optimization objective/gradient used by the
Frank-Wolfe loop (fw.frank_wolfe) and the LP oracle (fw.lp_oracle).
"""

import numpy as np
from numba import njit, prange

from fw.shattering import is_shattered_triple


@njit(parallel=True)
def _weighted_J(points, weights):
    m = points.shape[0]
    total = 0.0
    for i in prange(m):
        local = 0.0
        wi = weights[i]
        for j in range(m):
            wij = wi * weights[j]
            for k in range(m):
                if is_shattered_triple(points[i], points[j], points[k]):
                    local += wij * weights[k]
        total += local
    return total


@njit(parallel=True)
def _gradient_at_candidates(support_points, support_weights, candidate_points):
    m = support_points.shape[0]
    c = candidate_points.shape[0]
    grad = np.zeros(c, dtype=np.float64)
    for ci in prange(c):
        p = candidate_points[ci]
        acc = 0.0
        for q in range(m):
            wq = support_weights[q]
            for r in range(m):
                if is_shattered_triple(p, support_points[q], support_points[r]):
                    acc += wq * support_weights[r]
        grad[ci] = 3.0 * acc
    return grad


def compute_J(measure):
    """Exact J(measure), the fraction of shattered ordered triples under the
    measure (matches the brute-force computation in the legacy rec_AI.py, but
    generalized to arbitrary nonuniform weights)."""
    return float(_weighted_J(measure.points, measure.weights))


@njit(parallel=True)
def _trilinear_J(points1, weights1, points2, weights2, points3, weights3):
    m1 = points1.shape[0]
    total = 0.0
    for i in prange(m1):
        wi = weights1[i]
        local = 0.0
        for j in range(points2.shape[0]):
            wij = wi * weights2[j]
            for k in range(points3.shape[0]):
                if is_shattered_triple(points1[i], points2[j], points3[k]):
                    local += wij * weights3[k]
        total += local
    return total


def compute_trilinear_J(m1, m2, m3):
    """J(mu1, mu2, mu3) as defined in measure.tex Def. 2.2 -- the trilinear
    form whose diagonal (m1=m2=m3) is compute_J. Used by the Frank-Wolfe line
    search, which needs J along a segment (1-t) a + t o expanded via
    multilinearity rather than merging measures and recomputing J from
    scratch at every candidate t."""
    return float(
        _trilinear_J(m1.points, m1.weights, m2.points, m2.weights, m3.points, m3.weights)
    )


def compute_gradient(measure, candidate_points):
    """Gradient of J at `measure`, evaluated only at `candidate_points`
    (shape (C, 6)) -- the restricted-candidate-pool approximation described
    in measure.tex Sec. 5 step 2, since evaluating it at all n^6 cells is
    intractable. Returns a float64 array of shape (C,)."""
    candidate_points = np.asarray(candidate_points, dtype=np.int64)
    return _gradient_at_candidates(measure.points, measure.weights, candidate_points)
