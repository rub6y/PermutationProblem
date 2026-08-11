"""Direct empirical test of Conjecture "codim-1 concentration on H"
(measure.tex Sec.4) at the one size where we have the *exact* global
optimum in hand: n=4, S(4)=1 (all C(4,3)=4 triples shattered
simultaneously -- see 06_brute_force_small_n.py).

For every maximizing 6-tuple (WLOG sigma_1 = id), check whether it lies on
discrete H: row sums sum_k sigma_k(i) constant across all i (the discrete
analogue of mu(H)=1). Tally on-H vs off-H maximizers and print a few
off-H examples if any exist -- a single verified off-H global maximizer,
at any n, would be a genuine counterexample to Conjecture "codim 1" (not
just inconclusive absence of evidence, since the conjecture is a claim
about *every* maximizing mu, not just typical/found-by-search ones).
"""
import itertools
import sys

import numpy as np
from numba import njit, prange

sys.path.insert(0, "/home/ruba/Documents/Math/PermutationProblem")
from fw.shattering import is_shattered_triple  # noqa: E402


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
def scan_n4(perms):
    P = perms.shape[0]
    n = perms.shape[1]
    # per-a tallies: [num_maximizers, num_on_H]
    on_H = np.zeros(P, dtype=np.int64)
    off_H = np.zeros(P, dtype=np.int64)
    for a in prange(P):
        points = np.empty((n, 6), dtype=np.int64)
        points[:, 0] = np.arange(n)
        points[:, 1] = perms[a]
        local_on = 0
        local_off = 0
        for b in range(P):
            points[:, 2] = perms[b]
            for c in range(P):
                points[:, 3] = perms[c]
                for d in range(P):
                    points[:, 4] = perms[d]
                    for e in range(P):
                        points[:, 5] = perms[e]
                        t = _tri_count(points)
                        if t == 24:  # 4 * 6, all triples shattered
                            row_sums = points[:, 0] + points[:, 1] + points[:, 2] \
                                + points[:, 3] + points[:, 4] + points[:, 5]
                            on_h = True
                            base = row_sums[0]
                            for i in range(1, n):
                                if row_sums[i] != base:
                                    on_h = False
                                    break
                            if on_h:
                                local_on += 1
                            else:
                                local_off += 1
        on_H[a] = local_on
        off_H[a] = local_off
    return on_H.sum(), off_H.sum()


def first_off_H_example(perms):
    """Non-numba re-scan, stopping at the first off-H maximizer found, to
    print a concrete counterexample (or confirm none exists)."""
    P = perms.shape[0]
    n = perms.shape[1]
    for a in range(P):
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
                        if t == 24:
                            row_sums = points.sum(axis=1)
                            if len(set(row_sums.tolist())) > 1:
                                return points.copy()
    return None


if __name__ == "__main__":
    perms4 = np.array(list(itertools.permutations(range(4))), dtype=np.int64)
    on_H, off_H = scan_n4(perms4)
    total = on_H + off_H
    print(f"n=4 global maximizers (S(4)=1, WLOG sigma_1=id): {total} total")
    print(f"  on discrete H  (row sums constant): {on_H}")
    print(f"  off discrete H (row sums vary):      {off_H}")
    if off_H > 0:
        print("\nSearching for a concrete off-H maximizer to print...")
        ex = first_off_H_example(perms4)
        if ex is not None:
            print("Example off-H global maximizer (rows = points, cols = sigma_1..sigma_6):")
            print(ex)
            print("row sums:", ex.sum(axis=1))
