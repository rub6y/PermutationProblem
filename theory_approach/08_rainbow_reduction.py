"""Push on the reverse-pairing reduction from 07/last session's note:

Under sigma_4,5,6 = reverses of sigma_1,2,3, a triple is shattered iff
sigma_1,2,3's order-types on it fall into three different reversal-classes
of S_3: class0={123,321} (monotone triples), class1={132,231},
class2={213,312} (order_code -> class map below).

WLOG sigma_1 = id (same domain-relabeling argument as before). Then
sigma_1's order-type on *every* triple x<y<z is code 0 (identity pattern),
i.e. class 0, always. So "shattered" reduces to:

    class(sigma_2) != 0  AND  class(sigma_3) != 0  AND  class(sigma_2) != class(sigma_3)

i.e. a 2-permutation problem: maximize the fraction of triples where sigma_2
and sigma_3, compared to the identity, are both "non-monotone" (avoid being
class 0 = avoid triples that are themselves increasing or decreasing) and
land in *different* non-monotone classes.

Key observation before any search: class(sigma)=0 means the triple is
monotone (pattern 123 or 321) under sigma. By Erdos-Szekeres, no
permutation of length > 4 can avoid all monotone triples (avoiding length-3
increasing AND length-3 decreasing subsequences caps length at
(3-1)(3-1)=4) -- so for n>=5, class-0 density is *strictly* bounded away
from 0 for every permutation, for a combinatorial reason independent of
search. This script measures how small that density can actually be made
(minimum monotone-triple density) and what joint rainbow fraction is
achievable for sigma_2, sigma_3 together, as n grows.
"""
import sys
import time

import numpy as np
from numba import njit, prange

sys.path.insert(0, "/home/ruba/Documents/Math/PermutationProblem")
from fw.shattering import order_code  # noqa: E402

CLASS_OF_CODE = np.array([0, 1, 2, 1, 2, 0], dtype=np.int64)


@njit
def monotone_density(perm):
    """Fraction of triples with class 0 (monotone: 123 or 321) under perm
    vs. the identity reference order."""
    n = perm.shape[0]
    total = 0
    mono = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                code = order_code(perm[i], perm[j], perm[k])
                total += 1
                if code == 0 or code == 5:
                    mono += 1
    return mono / total


@njit
def rainbow_fraction(perm2, perm3):
    n = perm2.shape[0]
    total = 0
    rainbow = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                c2 = CLASS_OF_CODE[order_code(perm2[i], perm2[j], perm2[k])]
                c3 = CLASS_OF_CODE[order_code(perm3[i], perm3[j], perm3[k])]
                total += 1
                if c2 != 0 and c3 != 0 and c2 != c3:
                    rainbow += 1
    return rainbow / total


@njit
def anneal_monotone_min(n, seed, steps):
    """Simulated annealing to *minimize* monotone-triple density of a single
    permutation of length n (vs identity)."""
    np.random.seed(seed)
    perm = np.arange(n)
    np.random.shuffle(perm)
    cur = monotone_density(perm)
    best = cur
    best_perm = perm.copy()
    T0 = 0.05
    for s in range(steps):
        T = T0 * (1.0 - s / steps) + 1e-6
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i == j:
            continue
        perm[i], perm[j] = perm[j], perm[i]
        new = monotone_density(perm)
        if new <= cur or np.random.random() < np.exp(-(new - cur) / T):
            cur = new
            if cur < best:
                best = cur
                best_perm = perm.copy()
        else:
            perm[i], perm[j] = perm[j], perm[i]
    return best, best_perm


@njit
def anneal_rainbow_max(n, seed, steps):
    """Joint SA over (perm2, perm3) maximizing rainbow_fraction."""
    np.random.seed(seed)
    p2 = np.arange(n)
    np.random.shuffle(p2)
    p3 = np.arange(n)
    np.random.shuffle(p3)
    cur = rainbow_fraction(p2, p3)
    best = cur
    T0 = 0.05
    for s in range(steps):
        T = T0 * (1.0 - s / steps) + 1e-6
        which = np.random.randint(0, 2)
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i == j:
            continue
        if which == 0:
            p2[i], p2[j] = p2[j], p2[i]
        else:
            p3[i], p3[j] = p3[j], p3[i]
        new = rainbow_fraction(p2, p3)
        if new >= cur or np.random.random() < np.exp(-(cur - new) / T):
            cur = new
            if cur > best:
                best = cur
        else:
            if which == 0:
                p2[i], p2[j] = p2[j], p2[i]
            else:
                p3[i], p3[j] = p3[j], p3[i]
    return best


def layered_permutation(n, block):
    """Classical extremal construction for minimizing monotone triples:
    split [0,n) into ceil(n/block) blocks, each block internally
    *decreasing*, blocks arranged in *increasing* order of value ranges,
    alternated -- a 'layered' permutation. Block size ~ sqrt(n) is the
    Erdos-Szekeres-motivated choice (balances increasing-subsequence length
    across blocks against decreasing-subsequence length within a block)."""
    perm = []
    vals = list(range(n))
    for start in range(0, n, block):
        chunk = vals[start:start + block]
        perm.extend(reversed(chunk))
    return np.array(perm, dtype=np.int64)


if __name__ == "__main__":
    print("=== minimum monotone-triple density (single permutation vs id) ===")
    for n in [10, 20, 40, 80, 160, 320]:
        best_bal = 1.0
        for block in [max(1, int(round(n ** 0.5))) - 1, int(round(n ** 0.5)), int(round(n ** 0.5)) + 1]:
            if block < 1:
                continue
            p = layered_permutation(n, block)
            d = monotone_density(p)
            best_bal = min(best_bal, d)
        t0 = time.time()
        sa_best, sa_perm = anneal_monotone_min(n, 0, 400_000 if n <= 80 else 150_000)
        print(f"n={n:4d}: layered(block~sqrt n)={best_bal:.4f}  SA={sa_best:.4f}"
              f"  (SA {time.time()-t0:.1f}s)")

    print("\n=== joint rainbow fraction (sigma_2, sigma_3), SA ===")
    for n in [10, 20, 40, 80, 160]:
        t0 = time.time()
        best = anneal_rainbow_max(n, 0, 400_000 if n <= 80 else 150_000)
        print(f"n={n:4d}: rainbow fraction ~= {best:.4f}  ({time.time()-t0:.1f}s)")
