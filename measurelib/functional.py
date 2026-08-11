"""The J functional (measure.tex Def. 2.2): a symmetric cubic functional on
A that integrates the "triple is shattered" indicator against mu^{x3}.

J(mu) = sum_{i,j,k in support} w_i w_j w_k * shattered(i,j,k). Since
`shattered` is symmetric in its three arguments, J is a genuine cubic form
and its trilinear polarization J3(mu1, mu2, mu3) (diagonal = J) is exactly
the object measure.tex's necessary-condition-for-maximizers uses:
J(mu, mu, nu) <= J(mu, mu, mu) at a maximizer mu.

Independent implementation of the shattering predicate from fw/shattering.py
(cross-checked against it in measurelib/tests), so a bug in one is unlikely
to be masked by the other.
"""

import numpy as np
from numba import njit, prange

N_COORDS = 6


@njit(inline="always")
def _order_code(a, b, c):
    """Strict order type of three distinct values -> a code in [0, 6)."""
    if a < b:
        if b < c:
            return 0
        elif a < c:
            return 1
        else:
            return 3
    else:
        if a < c:
            return 2
        elif b < c:
            return 4
        else:
            return 5


@njit(inline="always")
def shattered(x1, x2, x3):
    """True if length-6 int arrays x1, x2, x3 induce all 6 distinct orderings
    across their 6 coordinates -- the "shattered triple" predicate."""
    seen_mask = 0
    unique_count = 0
    for d in range(N_COORDS):
        v1, v2, v3 = x1[d], x2[d], x3[d]
        if v1 == v2 or v1 == v3 or v2 == v3:
            return False

        bit = 1 << _order_code(v1, v2, v3)
        if (seen_mask & bit) == 0:
            seen_mask |= bit
            unique_count += 1

        if unique_count < d + 1:
            return False

    return unique_count == N_COORDS


@njit(parallel=True)
def _J3(points1, weights1, points2, weights2, points3, weights3):
    m1 = points1.shape[0]
    total = 0.0
    for i in prange(m1):
        wi = weights1[i]
        local = 0.0
        for j in range(points2.shape[0]):
            wij = wi * weights2[j]
            for k in range(points3.shape[0]):
                if shattered(points1[i], points2[j], points3[k]):
                    local += wij * weights3[k]
        total += local
    return total


@njit(parallel=True)
def _gradient_at_points(support_points, support_weights, at_points):
    m = support_points.shape[0]
    c = at_points.shape[0]
    grad = np.zeros(c, dtype=np.float64)
    for ci in prange(c):
        p = at_points[ci]
        acc = 0.0
        for q in range(m):
            wq = support_weights[q]
            for r in range(m):
                if shattered(p, support_points[q], support_points[r]):
                    acc += wq * support_weights[r]
        grad[ci] = 3.0 * acc
    return grad


def J(mu):
    """Exact J(mu, mu, mu): the fraction of mu^{x3}-mass on shattered triples."""
    return float(_J3(mu.points, mu.weights, mu.points, mu.weights, mu.points, mu.weights))


def J3(mu1, mu2, mu3):
    """The trilinear form J(mu1, mu2, mu3) (measure.tex Def. 2.2); diagonal
    (mu1=mu2=mu3) is J(mu). Used to check the maximizer necessary condition
    J(mu, mu, nu) <= J(mu, mu, mu), and for exact line search along segments."""
    return float(_J3(mu1.points, mu1.weights, mu2.points, mu2.weights, mu3.points, mu3.weights))


def gradient(mu, at_points):
    """Gradient of J at `mu`, evaluated only at `at_points` (shape (C, 6)):
    d/dw_p J = 3 * sum_{q,r in support} w_q w_r shattered(p,q,r). Evaluating
    at a restricted point set rather than the full n^6 grid is what makes
    gradient-based search over A_n tractable."""
    at_points = np.asarray(at_points, dtype=np.int64)
    return _gradient_at_points(mu.points, mu.weights, at_points)
