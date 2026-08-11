"""Candidate-pool generation for the restricted LP oracle (fw.lp_oracle).

Solving the true LP of measure.tex Sec. 5 step 3 exactly would require a
column per grid cell in [n]^6 (n^6 = ~3.1e8 for n=26), which is intractable.
Instead we restrict the LP to a pool of candidate grid points: the current
support (so the LP is always feasible -- the current measure itself is one
feasible solution), plus randomly sampled points, plus coordinatewise
perturbations ("neighbors") of the current support. This is the one place
the search strategy lives, so it can be tuned independently of the LP oracle
and the Frank-Wolfe loop.

Gradient evaluation at a candidate point costs O(support_size^2), so for the
support sizes this search actually deals with (tens to low hundreds) it's
cheap to evaluate every single-coordinate move of every support point exactly,
rather than randomly subsampling that neighborhood: previously
`neighbor_points` only tried `n_neighbors_per_point` random shifts within a
small `radius`, which could (and empirically did) miss every improving
direction even when one existed one axis-move away.
"""

import numpy as np

N_COORDS = 6


def random_points(n, count, rng):
    return rng.integers(0, n, size=(count, N_COORDS), dtype=np.int64)


def full_neighbor_points(points, n):
    """For each row in `points` and each coordinate, every variant obtained
    by replacing that one coordinate with every value in [0, n) (including
    the original, which is harmless since the pool is deduplicated)."""
    m = points.shape[0]
    out = np.repeat(points, N_COORDS * n, axis=0)
    coord = np.tile(np.repeat(np.arange(N_COORDS), n), m)
    value = np.tile(np.arange(n), m * N_COORDS)
    rows = np.arange(m * N_COORDS * n)
    out[rows, coord] = value
    return out


def build_candidate_pool(measure, n_random=400, seed=None):
    """Union of: current support (guarantees LP feasibility), fresh random
    points, and every single-coordinate move of the current support. Duplicate
    rows are removed."""
    rng = np.random.default_rng(seed)
    n = measure.n
    parts = [
        measure.points,
        random_points(n, n_random, rng),
        full_neighbor_points(measure.points, n),
    ]
    pool = np.concatenate(parts, axis=0)
    return np.unique(pool, axis=0)
