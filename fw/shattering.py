"""Single source of truth for the "shattered triple" predicate.

Three grid points (each a length-6 tuple of coordinates) are shattered when
their six coordinatewise 3-element orderings are pairwise distinct, i.e. they
realize all 6 elements of S_3. This module is the only place that predicate
is implemented; every other module (objective, legacy scripts) must import
it rather than reimplementing the order-code logic.
"""

import numpy as np
from numba import njit, prange

N_COORDS = 6


@njit(inline="always")
def order_code(a, b, c):
    """Map the strict order type of three distinct values to a code in [0, 6)."""
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
def is_shattered_triple(x1, x2, x3):
    """True if length-6 int arrays x1, x2, x3 are ordered differently in all 6 coords."""
    seen_mask = 0
    unique_count = 0
    for d in range(N_COORDS):
        v1, v2, v3 = x1[d], x2[d], x3[d]
        if v1 == v2 or v1 == v3 or v2 == v3:
            return False

        bit = 1 << order_code(v1, v2, v3)
        if (seen_mask & bit) == 0:
            seen_mask |= bit
            unique_count += 1

        if unique_count < d + 1:
            return False

    return unique_count == N_COORDS


@njit(parallel=True)
def count_shattered_triples(points):
    """Number of ordered triples (i, j, k) of rows in `points` that are shattered."""
    m = points.shape[0]
    total = 0
    for i in prange(m):
        local = 0
        for j in range(m):
            for k in range(m):
                if is_shattered_triple(points[i], points[j], points[k]):
                    local += 1
        total += local
    return total


def shattering_reference(x1, x2, x3):
    """Slow, pure-Python/argsort-based implementation of the same predicate.
    Used to regression-test is_shattered_triple, and by legacy scripts that
    operate on float coordinates (numba's int-typed order_code path is not
    used there)."""
    orderings = set()
    for d in range(N_COORDS):
        v1, v2, v3 = x1[d], x2[d], x3[d]
        if v1 == v2 or v1 == v3 or v2 == v3:
            return False
        ordering = tuple(np.argsort(np.argsort(np.array([v1, v2, v3]))))
        orderings.add(ordering)
        if len(orderings) < d + 1:
            return False
    return len(orderings) == 6
