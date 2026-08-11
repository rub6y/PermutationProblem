"""Exact small-n brute force for S(n), plus a check of the pairwise-balance
necessary condition derived by hand (see 05b_pairwise_balance.md):
if {x,y,z} is shattered, then for every pair inside it, exactly 3 of the 6
coordinates put that pair one way and 3 the other.

WLOG sigma_1 = identity: T_n(sigma_1,...,sigma_6) = T_n(sigma_1 o pi, ...,
sigma_6 o pi) for any pi in S_n (simultaneous domain relabeling doesn't
change which unordered triples are shattered), so take pi = sigma_1^{-1}.
This cuts the search space by a factor of n!.

n=4: exhaustive over the remaining 5 coordinates (24^5 ~ 8M), exact.
n=5,6,7: exhaustive is 120^5+ -- infeasible; random search instead (many
independent random 5-tuples of permutations), reported as a lower bound /
approximate optimum, not exact.
"""
import itertools
import sys
import time

import numpy as np
from numba import njit, prange

sys.path.insert(0, "/home/ruba/Documents/Math/PermutationProblem")
from fw.shattering import is_shattered_triple, count_shattered_triples  # noqa: E402


@njit
def _tri_count(points):
    m = points.shape[0]
    total = 0
    for i in range(m):
        for j in range(m):
            for k in range(m):
                if is_shattered_triple(points[i], points[j], points[k]):
                    total += 1
    return total


@njit(parallel=True)
def brute_force_n4(perms):
    # perms: (24, 4) all permutations of range(4)
    P = perms.shape[0]
    n = perms.shape[1]
    best = np.zeros(P, dtype=np.int64)
    for a in prange(P):
        local_best = 0
        points = np.empty((n, 6), dtype=np.int64)
        points[:, 0] = np.arange(n)
        points[:, 1] = perms[a]
        for b in range(P):
            points[:, 2] = perms[b]
            for c in range(P):
                points[:, 3] = perms[c]
                for d in range(P):
                    points[:, 4] = perms[d]
                    for e in range(P):
                        points[:, 5] = perms[e]
                        t = _tri_count(points)
                        if t > local_best:
                            local_best = t
        best[a] = local_best
    return best.max() // 6


def binom3(n):
    return n * (n - 1) * (n - 2) // 6


def pairwise_balance_check(points):
    """Return True iff every pair that appears in some shattered triple has
    b(x,y) == 3 (3 of the 6 coordinates agree x<y, 3 disagree)."""
    n = points.shape[0]
    ok = True
    for x in range(n):
        for y in range(x + 1, n):
            in_shattered = False
            for z in range(n):
                if z in (x, y):
                    continue
                a, b, c = sorted([x, y, z])
                if is_shattered_triple(points[a], points[b], points[c]):
                    in_shattered = True
                    break
            if in_shattered:
                b_count = sum(1 for k in range(6) if points[x][k] < points[y][k])
                if b_count != 3:
                    ok = False
    return ok


def random_search(n, trials, rng):
    best_T = 0
    best_points = None
    ident = np.arange(n)
    for _ in range(trials):
        points = np.empty((n, 6), dtype=np.int64)
        points[:, 0] = ident
        for c in range(1, 6):
            points[:, c] = rng.permutation(n)
        t = count_shattered_triples(points) // 6
        if t > best_T:
            best_T = t
            best_points = points.copy()
    return best_T, best_points


if __name__ == "__main__":
    print("=== n=4 exhaustive (WLOG sigma_1 = id) ===")
    perms4 = np.array(list(itertools.permutations(range(4))), dtype=np.int64)
    t0 = time.time()
    best4 = brute_force_n4(perms4)
    print(f"max T_4 = {best4} / C(4,3) = {binom3(4)}  ->  S(4) = {best4 / binom3(4):.6f}"
          f"  ({time.time()-t0:.1f}s)")

    print("\n=== n=5..8 random search (approximate) ===")
    rng = np.random.default_rng(0)
    for n, trials in [(5, 200_000), (6, 200_000), (7, 150_000), (8, 100_000)]:
        t0 = time.time()
        best_T, best_points = random_search(n, trials, rng)
        frac = best_T / binom3(n)
        balance_ok = pairwise_balance_check(best_points)
        print(f"n={n}: best T={best_T}/{binom3(n)} -> S~={frac:.6f}  "
              f"pairwise-balance holds on best: {balance_ok}  ({time.time()-t0:.1f}s)")
