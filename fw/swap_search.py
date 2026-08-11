"""Block-swap local search directly on the permutation representation.

fw.frank_wolfe's LP oracle only ever proposes a single new *vertex* of A_n and
takes a linear step towards it, so it can find no improving direction once a
configuration is optimal against every single-point relocation -- confirmed
for good_permutation.txt even with the full one-axis-neighbor candidate pool
(fw.candidates.full_neighbor_points): the LP oracle returns the current
support unchanged.

good_permutation.txt has support size n == 26, i.e. it is literally 6
permutations of [n] (fw.measure.measure_from_permutations): every point
carries equal weight 1/n and every coordinate is a bijection [n] -> [n]. For
this structure there's a move that stays inside A_n for free, with no LP
needed: pick one coordinate k and transpose the values at two positions i, j
(equivalently, compose sigma_k with a transposition) -- this can reposition
several grid points relative to each other in one step, unlike a single-vertex
FW move. This module searches that neighborhood exhaustively, and wraps it in
basin-hopping (hill-climb to a local optimum, perturb with random swaps,
repeat) so multi-swap escapes from 1-swap-local optima are possible too.
"""

from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange

from fw.shattering import count_shattered_triples, is_shattered_triple

N_COORDS = 6


def exact_J(points):
    """J = fraction of shattered ordered triples, for an equal-weight,
    all-distinct-points measure (points.shape == (n, 6))."""
    n = points.shape[0]
    return count_shattered_triples(points) / n**3


@njit(inline="always")
def _swap_delta(points, coord, i, j):
    """Change in shattered-triple *count* from transposing points[i, coord]
    and points[j, coord]. A triple's shattered status can only change if it
    contains row i or row j (every other row is untouched), so we only need
    to rescan the O(n^2) triples touching {i, j} instead of all O(n^3)
    triples -- for n=26 that's 3752 vs 17576 triples per candidate swap."""
    n = points.shape[0]
    old_pi = points[i, coord]
    old_pj = points[j, coord]

    before = 0
    # block A: p in {i, j}, q, r free
    for p in (i, j):
        for q in range(n):
            for r in range(n):
                if is_shattered_triple(points[p], points[q], points[r]):
                    before += 1
    # block B: p not in {i, j}, q in {i, j}, r free
    for p in range(n):
        if p == i or p == j:
            continue
        for q in (i, j):
            for r in range(n):
                if is_shattered_triple(points[p], points[q], points[r]):
                    before += 1
    # block C: p, q not in {i, j}, r in {i, j}
    for p in range(n):
        if p == i or p == j:
            continue
        for q in range(n):
            if q == i or q == j:
                continue
            for r in (i, j):
                if is_shattered_triple(points[p], points[q], points[r]):
                    before += 1

    points[i, coord] = old_pj
    points[j, coord] = old_pi

    after = 0
    for p in (i, j):
        for q in range(n):
            for r in range(n):
                if is_shattered_triple(points[p], points[q], points[r]):
                    after += 1
    for p in range(n):
        if p == i or p == j:
            continue
        for q in (i, j):
            for r in range(n):
                if is_shattered_triple(points[p], points[q], points[r]):
                    after += 1
    for p in range(n):
        if p == i or p == j:
            continue
        for q in range(n):
            if q == i or q == j:
                continue
            for r in (i, j):
                if is_shattered_triple(points[p], points[q], points[r]):
                    after += 1

    points[i, coord] = old_pi
    points[j, coord] = old_pj

    return after - before


@njit(parallel=True)
def _best_swap(points):
    """Exhaustively try every (coord, i, j) transposition and return the one
    with the largest increase in shattered-triple count, plus its delta.
    delta <= 0 means no swap improves on the current configuration."""
    n = points.shape[0]
    n_moves = N_COORDS * n * (n - 1) // 2
    deltas = np.empty(n_moves, dtype=np.int64)
    coords = np.empty(n_moves, dtype=np.int64)
    is_ = np.empty(n_moves, dtype=np.int64)
    js_ = np.empty(n_moves, dtype=np.int64)

    move = 0
    for coord in range(N_COORDS):
        for i in range(n):
            for j in range(i + 1, n):
                coords[move] = coord
                is_[move] = i
                js_[move] = j
                move += 1

    for m in prange(n_moves):
        # each thread mutates its own private copy so parallel iterations
        # (which temporarily swap-and-revert in place) don't clash
        local_pts = points.copy()
        deltas[m] = _swap_delta(local_pts, coords[m], is_[m], js_[m])

    best_idx = np.argmax(deltas)
    return coords[best_idx], is_[best_idx], js_[best_idx], deltas[best_idx]


def apply_swap(points, coord, i, j):
    out = points.copy()
    out[i, coord], out[j, coord] = out[j, coord], out[i, coord]
    return out


def hill_climb(points, max_steps=1000):
    """Repeatedly apply the best-improving transposition until none improves.
    Returns (points, J, n_steps)."""
    points = points.copy()
    n = points.shape[0]
    steps = 0
    while steps < max_steps:
        coord, i, j, delta = _best_swap(points)
        if delta <= 0:
            break
        points = apply_swap(points, coord, i, j)
        steps += 1
    return points, count_shattered_triples(points) / n**3, steps


@dataclass
class BasinHopResult:
    best_points: np.ndarray
    best_J: float
    history: list = field(default_factory=list)


def basin_hop(points, n_restarts=20, perturb_swaps=3, seed=None, max_hill_climb_steps=1000):
    """Hill-climb to a local optimum, then repeatedly: perturb with
    `perturb_swaps` random transpositions and hill-climb again, keeping the
    best configuration found across all restarts (simple basin-hopping)."""
    rng = np.random.default_rng(seed)
    n = points.shape[0]

    best_points, best_J, _ = hill_climb(points, max_hill_climb_steps)
    history = [best_J]

    for _ in range(n_restarts):
        # always perturb from the best point found so far, not from wherever
        # the last walk ended -- otherwise a run of unlucky restarts drifts
        # the search permanently away from the good basin (observed: greedy
        # hill-climb from a heavily perturbed state can land somewhere worse
        # than where it started, and there's nothing pulling it back)
        perturbed = best_points.copy()
        for _ in range(perturb_swaps):
            coord = rng.integers(0, N_COORDS)
            i, j = rng.choice(n, size=2, replace=False)
            perturbed = apply_swap(perturbed, coord, i, j)

        candidate, candidate_J, _ = hill_climb(perturbed, max_hill_climb_steps)
        history.append(candidate_J)

        if candidate_J > best_J:
            best_points, best_J = candidate, candidate_J

    return BasinHopResult(best_points=best_points, best_J=best_J, history=history)
