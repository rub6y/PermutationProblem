"""Linear-programming oracle: step 3 of the Frank-Wolfe recipe in
measure.tex Sec. 5 -- find v in W maximizing <grad, v>, restricted to a
candidate pool of grid points (fw.candidates) rather than the full [n]^6
grid, since W has n^6 columns for the true problem.

W = { v >= 0 : marginal(v, coord) == 1/n for every coord in [6] }, i.e. the
6-way analogue of a transportation polytope. Restricting the LP to a
candidate pool means only supplying columns for those grid cells; the
current measure's own support is always included in the pool (fw.candidates)
so the LP is guaranteed feasible.
"""

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from fw.measure import Measure, N_COORDS


def _marginal_constraint_matrix(pool_points, n):
    """Sparse (6n, C) matrix A such that (A v)[coord * n + i] = sum of v_c
    over candidates c landing in grid value i along `coord`."""
    c = pool_points.shape[0]
    rows = np.empty(N_COORDS * c, dtype=np.int64)
    cols = np.empty(N_COORDS * c, dtype=np.int64)
    for coord in range(N_COORDS):
        rows[coord * c:(coord + 1) * c] = coord * n + pool_points[:, coord]
        cols[coord * c:(coord + 1) * c] = np.arange(c)
    data = np.ones(N_COORDS * c, dtype=np.float64)
    return sparse.csr_matrix((data, (rows, cols)), shape=(N_COORDS * n, c))


def solve_lp_oracle(pool_points, gradient, n):
    """Return the LP-optimal vertex o in W (restricted to `pool_points`) that
    maximizes <gradient, o>, as a Measure. Drops zero-weight columns."""
    a_eq = _marginal_constraint_matrix(pool_points, n)
    b_eq = np.full(N_COORDS * n, 1.0 / n, dtype=np.float64)

    result = linprog(
        c=-gradient,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=(0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"LP oracle failed: {result.message}")

    weights = result.x
    keep = weights > 1e-12
    return Measure(points=pool_points[keep], weights=weights[keep], n=n)
