"""Same rainbow-fraction question as 08_rainbow_reduction.py, but with
O(n^2)-per-step incremental SA updates instead of recomputing the full
O(n^3) triple sum after every swap -- 08's SA was diagnosed mid-run as
hitting exactly the scaling mistake track1_scaling_diagnosis.md flagged
earlier this project (fixed step budget, but each step's cost grows with
n, and here it grew as badly as possible, O(n^3)). A position swap only
changes the codes of triples that include one of the two swapped
positions -- O(n^2) of them -- so only those need to be re-scored.
"""
import sys
import time

import numpy as np
from numba import njit

sys.path.insert(0, "/home/ruba/Documents/Math/PermutationProblem")
from fw.shattering import order_code  # noqa: E402

CLASS_OF_CODE = np.array([0, 1, 2, 1, 2, 0], dtype=np.int64)


@njit
def monotone_density_full(perm):
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
    return mono, total


@njit
def touching_mono_count(perm, i, j):
    """Sum of the monotone indicator over every triple containing position
    i and/or j (i < j), i.e. exactly the triples whose code changes when
    perm[i], perm[j] are swapped."""
    n = perm.shape[0]
    total = 0
    for p in range(n):
        if p == i or p == j:
            continue
        a, b, c = sorted((i, j, p))
        code = order_code(perm[a], perm[b], perm[c])
        if code == 0 or code == 5:
            total += 1
    for p in range(n):
        if p == i or p == j:
            continue
        for q in range(p + 1, n):
            if q == i or q == j:
                continue
            a, b, c = sorted((i, p, q))
            code = order_code(perm[a], perm[b], perm[c])
            if code == 0 or code == 5:
                total += 1
            a, b, c = sorted((j, p, q))
            code = order_code(perm[a], perm[b], perm[c])
            if code == 0 or code == 5:
                total += 1
    return total


@njit
def anneal_monotone_min(n, seed, steps, T0):
    np.random.seed(seed)
    perm = np.arange(n)
    np.random.shuffle(perm)
    mono, total = monotone_density_full(perm)
    best = mono / total
    for s in range(steps):
        T = T0 * (1.0 - s / steps) + 1e-6
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        before = touching_mono_count(perm, i, j)
        perm[i], perm[j] = perm[j], perm[i]
        after = touching_mono_count(perm, i, j)
        new_mono = mono - before + after
        new_density = new_mono / total
        cur_density = mono / total
        if new_density <= cur_density or np.random.random() < np.exp(-(new_density - cur_density) / T):
            mono = new_mono
            if new_density < best:
                best = new_density
        else:
            perm[i], perm[j] = perm[j], perm[i]
    return best


@njit
def touching_rainbow_count(perm2, perm3, i, j):
    """Sum of the rainbow indicator (class(perm2)!=0, class(perm3)!=0,
    classes differ) over every triple containing position i and/or j."""
    n = perm2.shape[0]
    total = 0
    for p in range(n):
        if p == i or p == j:
            continue
        a, b, c = sorted((i, j, p))
        c2 = CLASS_OF_CODE[order_code(perm2[a], perm2[b], perm2[c])]
        c3 = CLASS_OF_CODE[order_code(perm3[a], perm3[b], perm3[c])]
        if c2 != 0 and c3 != 0 and c2 != c3:
            total += 1
    for p in range(n):
        if p == i or p == j:
            continue
        for q in range(p + 1, n):
            if q == i or q == j:
                continue
            for base in (i, j):
                a, b, c = sorted((base, p, q))
                c2 = CLASS_OF_CODE[order_code(perm2[a], perm2[b], perm2[c])]
                c3 = CLASS_OF_CODE[order_code(perm3[a], perm3[b], perm3[c])]
                if c2 != 0 and c3 != 0 and c2 != c3:
                    total += 1
    return total


@njit
def rainbow_full(perm2, perm3):
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
    return rainbow, total


@njit
def anneal_rainbow_max(n, seed, steps, T0):
    np.random.seed(seed)
    p2 = np.arange(n)
    np.random.shuffle(p2)
    p3 = np.arange(n)
    np.random.shuffle(p3)
    rb, total = rainbow_full(p2, p3)
    best = rb / total
    for s in range(steps):
        T = T0 * (1.0 - s / steps) + 1e-6
        which = np.random.randint(0, 2)
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        if which == 0:
            before = touching_rainbow_count(p2, p3, i, j)
            p2[i], p2[j] = p2[j], p2[i]
            after = touching_rainbow_count(p2, p3, i, j)
        else:
            before = touching_rainbow_count(p2, p3, i, j)
            p3[i], p3[j] = p3[j], p3[i]
            after = touching_rainbow_count(p2, p3, i, j)
        new_rb = rb - before + after
        new_frac = new_rb / total
        cur_frac = rb / total
        if new_frac >= cur_frac or np.random.random() < np.exp(-(cur_frac - new_frac) / T):
            rb = new_rb
            if new_frac > best:
                best = new_frac
        else:
            if which == 0:
                p2[i], p2[j] = p2[j], p2[i]
            else:
                p3[i], p3[j] = p3[j], p3[i]
    return best


def steps_for(n, base_steps_at_100=200_000):
    # per-step cost is O(n^2); keep step_count * n^2 roughly constant so
    # wall time per n is roughly flat instead of exploding.
    return max(20_000, int(base_steps_at_100 * (100.0 / n) ** 2))


if __name__ == "__main__":
    print("=== minimum monotone-triple density (single permutation vs id), incremental SA ===")
    for n in [10, 20, 40, 80, 160, 320]:
        t0 = time.time()
        steps = steps_for(n)
        best = min(anneal_monotone_min(n, r, steps, 0.05) for r in range(2))
        print(f"n={n:5d}: min monotone density ~= {best:.4f}  steps={steps}  ({time.time()-t0:.1f}s)", flush=True)

    print("\n=== joint rainbow fraction (sigma_2, sigma_3), incremental SA ===")
    for n in [10, 20, 40, 80, 160, 320]:
        t0 = time.time()
        steps = steps_for(n)
        best = max(anneal_rainbow_max(n, r, steps, 0.05) for r in range(2))
        print(f"n={n:5d}: rainbow fraction ~= {best:.4f}  steps={steps}  ({time.time()-t0:.1f}s)", flush=True)
