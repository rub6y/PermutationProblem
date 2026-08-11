"""Hyperplane-constrained variant of fw.swap_search.

scripts/explore_small_n.py found that every point of every near-optimal
6-tuple found so far (n=4..8 via fresh basin-hop, and n=26's
good_permutation.txt) satisfies sum_k sigma_k(i) == 3*(n-1) exactly -- the
6-D analogue of the pair-shattering problem's anti-diagonal maximizer
(measure.tex Obs 2.9). fw.swap_search's single-coordinate transposition
does *not* preserve this per-point row sum (swapping sigma_k(i),sigma_k(j)
changes both points' row sums, generically off the hyperplane), so an
unconstrained search wanders off the hyperplane and back.

The minimal move that changes two points' coordinates while leaving both
row sums exactly fixed is a *simultaneous* double transposition: pick two
distinct coordinates k, l and two points i, j, and swap
sigma_k(i)<->sigma_k(j) *and* sigma_l(i)<->sigma_l(j) at once. Point i's row
sum changes by (sigma_k(j)-sigma_k(i)) + (sigma_l(j)-sigma_l(i)); this is
zero exactly when sigma_k(i)+sigma_l(i) == sigma_k(j)+sigma_l(j) -- i.e. i
and j already agree on their partial sum over {k, l}. Restricting to such
(i, j) pairs keeps every point on the hyperplane for the entire search, not
just at a local optimum found by an unconstrained walk.
"""

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from numba import njit, prange

from fw.shattering import count_shattered_triples, is_shattered_triple

N_COORDS = 6


def exact_J(points):
    n = points.shape[0]
    return count_shattered_triples(points) / n**3


def row_sums_on_hyperplane(points):
    n = points.shape[0]
    target = N_COORDS * (n - 1) // 2 if (N_COORDS * (n - 1)) % 2 == 0 else None
    sums = points.sum(axis=1)
    return sums, (target is not None and np.all(sums == target))


@njit(inline="always")
def _double_swap_delta(points, k, l, i, j):
    """Change in shattered-triple count from the simultaneous double
    transposition (coord k and coord l, points i and j). Same O(n^2)
    reduced-rescan trick as fw.swap_search._swap_delta: only rows i, j
    change, so only triples touching {i, j} can flip status."""
    n = points.shape[0]
    old_ki, old_kj = points[i, k], points[j, k]
    old_li, old_lj = points[i, l], points[j, l]

    before = 0
    for p in (i, j):
        for q in range(n):
            for r in range(n):
                if is_shattered_triple(points[p], points[q], points[r]):
                    before += 1
    for p in range(n):
        if p == i or p == j:
            continue
        for q in (i, j):
            for r in range(n):
                if is_shattered_triple(points[p], points[q], points[r]):
                    before += 1
    for p in range(n):
        if p == i or p == j:
            continue
        for q in range(n):
            if q == i or q == j:
                continue
            for r in (i, j):
                if is_shattered_triple(points[p], points[q], points[r]):
                    before += 1

    points[i, k], points[j, k] = old_kj, old_ki
    points[i, l], points[j, l] = old_lj, old_li

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

    points[i, k], points[j, k] = old_ki, old_kj
    points[i, l], points[j, l] = old_li, old_lj

    return after - before


def apply_double_swap(points, k, l, i, j):
    out = points.copy()
    out[i, k], out[j, k] = out[j, k], out[i, k]
    out[i, l], out[j, l] = out[j, l], out[i, l]
    return out


def _candidate_moves(points):
    """All (k, l, i, j) with k < l and sigma_k(i)+sigma_l(i) ==
    sigma_k(j)+sigma_l(j) -- the hyperplane-preserving double swaps.
    Built in plain Python via grouping (cheap: O(n) per coordinate pair)
    rather than the O(n^2) all-pairs scan an unconstrained search would need."""
    n = points.shape[0]
    moves = []
    for k in range(N_COORDS):
        for l in range(k + 1, N_COORDS):
            groups = {}
            partial = points[:, k] + points[:, l]
            for idx, val in enumerate(partial):
                groups.setdefault(int(val), []).append(idx)
            for idxs in groups.values():
                if len(idxs) < 2:
                    continue
                for a in range(len(idxs)):
                    for b in range(a + 1, len(idxs)):
                        moves.append((k, l, idxs[a], idxs[b]))
    return moves


@njit(parallel=True)
def _evaluate_moves(points, ks, ls, is_, js_):
    n_moves = ks.shape[0]
    deltas = np.empty(n_moves, dtype=np.int64)
    for m in prange(n_moves):
        local_pts = points.copy()
        deltas[m] = _double_swap_delta(local_pts, ks[m], ls[m], is_[m], js_[m])
    return deltas


def _best_hyperplane_swap(points):
    moves = _candidate_moves(points)
    if not moves:
        return None
    ks = np.array([m[0] for m in moves], dtype=np.int64)
    ls = np.array([m[1] for m in moves], dtype=np.int64)
    is_ = np.array([m[2] for m in moves], dtype=np.int64)
    js_ = np.array([m[3] for m in moves], dtype=np.int64)
    deltas = _evaluate_moves(points, ks, ls, is_, js_)
    best_idx = np.argmax(deltas)
    return int(ks[best_idx]), int(ls[best_idx]), int(is_[best_idx]), int(js_[best_idx]), int(deltas[best_idx])


def hill_climb_hyperplane(points, max_steps=1000):
    """Like fw.swap_search.hill_climb, but restricted to moves that keep
    every point's row sum exactly on the central hyperplane throughout."""
    points = points.copy()
    n = points.shape[0]
    steps = 0
    while steps < max_steps:
        best = _best_hyperplane_swap(points)
        if best is None:
            break
        k, l, i, j, delta = best
        if delta <= 0:
            break
        points = apply_double_swap(points, k, l, i, j)
        steps += 1
    return points, count_shattered_triples(points) / n**3, steps


@dataclass
class BasinHopResult:
    best_points: np.ndarray
    best_J: float
    history: list = field(default_factory=list)


@njit(inline="always")
def _kcycle_delta(points, k_coord, l_coord, idxs):
    """Change in shattered-triple count from cyclically rotating the
    (k_coord, l_coord) value pairs among the m points in `idxs`: point
    idxs[t] receives idxs[t+1 mod m]'s (k_coord, l_coord) values. This
    generalizes the double-swap (m=2) to m=3,4,... points at once, and still
    preserves every touched row's sum exactly, since it's just a relabeling
    of value pairs within a group that already agrees on
    sigma_k(i)+sigma_l(i)."""
    n = points.shape[0]
    m = idxs.shape[0]
    touched = np.zeros(n, dtype=np.bool_)
    for t in range(m):
        touched[idxs[t]] = True

    old_k = np.empty(m, dtype=points.dtype)
    old_l = np.empty(m, dtype=points.dtype)
    for t in range(m):
        old_k[t] = points[idxs[t], k_coord]
        old_l[t] = points[idxs[t], l_coord]

    before = 0
    for t in range(m):
        p = idxs[t]
        for q in range(n):
            for r in range(n):
                if is_shattered_triple(points[p], points[q], points[r]):
                    before += 1
    for p in range(n):
        if touched[p]:
            continue
        for t in range(m):
            q = idxs[t]
            for r in range(n):
                if is_shattered_triple(points[p], points[q], points[r]):
                    before += 1
    for p in range(n):
        if touched[p]:
            continue
        for q in range(n):
            if touched[q]:
                continue
            for t in range(m):
                r = idxs[t]
                if is_shattered_triple(points[p], points[q], points[r]):
                    before += 1

    for t in range(m):
        nxt = (t + 1) % m
        points[idxs[t], k_coord] = old_k[nxt]
        points[idxs[t], l_coord] = old_l[nxt]

    after = 0
    for t in range(m):
        p = idxs[t]
        for q in range(n):
            for r in range(n):
                if is_shattered_triple(points[p], points[q], points[r]):
                    after += 1
    for p in range(n):
        if touched[p]:
            continue
        for t in range(m):
            q = idxs[t]
            for r in range(n):
                if is_shattered_triple(points[p], points[q], points[r]):
                    after += 1
    for p in range(n):
        if touched[p]:
            continue
        for q in range(n):
            if touched[q]:
                continue
            for t in range(m):
                r = idxs[t]
                if is_shattered_triple(points[p], points[q], points[r]):
                    after += 1

    for t in range(m):
        points[idxs[t], k_coord] = old_k[t]
        points[idxs[t], l_coord] = old_l[t]

    return after - before


def apply_kcycle(points, k_coord, l_coord, idxs):
    out = points.copy()
    m = len(idxs)
    old_k = [out[i, k_coord] for i in idxs]
    old_l = [out[i, l_coord] for i in idxs]
    for t in range(m):
        nxt = (t + 1) % m
        out[idxs[t], k_coord] = old_k[nxt]
        out[idxs[t], l_coord] = old_l[nxt]
    return out


def _candidate_k_cycles(points, k_size, max_per_group=200, rng=None):
    """All (k, l, idxs) where idxs is a k_size-subset of points that agree on
    sigma_k(i)+sigma_l(i) -- the hyperplane-preserving k-cycles. Groups
    larger than k_size are subsampled (max_per_group combinations) since the
    number of k-subsets grows combinatorially with group size."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = points.shape[0]
    moves = []
    for k in range(N_COORDS):
        for l in range(k + 1, N_COORDS):
            groups = {}
            partial = points[:, k] + points[:, l]
            for idx, val in enumerate(partial):
                groups.setdefault(int(val), []).append(idx)
            for idxs in groups.values():
                if len(idxs) < k_size:
                    continue
                combos = list(combinations(idxs, k_size))
                if len(combos) > max_per_group:
                    pick = rng.choice(len(combos), max_per_group, replace=False)
                    combos = [combos[i] for i in pick]
                for c in combos:
                    moves.append((k, l, np.array(c, dtype=np.int64)))
    return moves


@njit(parallel=True)
def _evaluate_kcycle_moves(points, ks, ls, idxs_matrix):
    n_moves = ks.shape[0]
    deltas = np.empty(n_moves, dtype=np.int64)
    for m in prange(n_moves):
        local_pts = points.copy()
        deltas[m] = _kcycle_delta(local_pts, ks[m], ls[m], idxs_matrix[m])
    return deltas


def basin_hop_hyperplane(points, n_restarts=20, perturb_swaps=3, seed=None, max_hill_climb_steps=1000):
    """Basin-hopping using only hyperplane-preserving double swaps, both for
    the hill-climb and for the perturbation step (so the whole walk, not
    just the reported optimum, stays on the hyperplane)."""
    rng = np.random.default_rng(seed)
    n = points.shape[0]

    best_points, best_J, _ = hill_climb_hyperplane(points, max_hill_climb_steps)
    history = [best_J]

    for _ in range(n_restarts):
        perturbed = best_points.copy()
        applied = 0
        attempts = 0
        while applied < perturb_swaps and attempts < perturb_swaps * 20:
            attempts += 1
            moves = _candidate_moves(perturbed)
            if not moves:
                break
            k, l, i, j = moves[rng.integers(0, len(moves))]
            perturbed = apply_double_swap(perturbed, k, l, i, j)
            applied += 1

        candidate, candidate_J, _ = hill_climb_hyperplane(perturbed, max_hill_climb_steps)
        history.append(candidate_J)

        if candidate_J > best_J:
            best_points, best_J = candidate, candidate_J

    return BasinHopResult(best_points=best_points, best_J=best_J, history=history)
