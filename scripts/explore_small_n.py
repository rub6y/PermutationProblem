"""Exploratory search: find near-optimal 6-tuples of permutations for small n
by random-restart + basin-hopping, then print structural features (column
sums, relation to identity/reversal, cycle structure, distances between the
six permutations) to look for patterns shared with good_permutation.txt
(n=26, J=482/975).

Usage: nix-shell shell.nix --run "python3 scripts/explore_small_n.py 4 5 6 7 8"
"""

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from fw.swap_search import basin_hop, hill_climb, exact_J

N_COORDS = 6


def random_points(n, rng):
    points = np.empty((n, N_COORDS), dtype=np.int64)
    for k in range(N_COORDS):
        points[:, k] = rng.permutation(n)
    return points


def search(n, n_starts=12, n_restarts=25, perturb_swaps=3, seed=0):
    rng = np.random.default_rng(seed)
    best_points, best_J = None, -1.0
    for s in range(n_starts):
        pts = random_points(n, rng)
        res = basin_hop(pts, n_restarts=n_restarts, perturb_swaps=perturb_swaps,
                         seed=int(rng.integers(0, 1 << 30)))
        if res.best_J > best_J:
            best_J, best_points = res.best_J, res.best_points
    return best_points, best_J


def describe(points, n):
    print(f"  n={n}  J={exact_J(points):.6f}  (2/n asymptotics: 1/2)")
    col_sums = points.sum(axis=0)
    print(f"  per-permutation sum of values (perm as function values): {col_sums.tolist()}")
    row_sums = points.sum(axis=1)
    print(f"  row sums (should cluster near {N_COORDS*(n-1)/2:.1f} if on central hyperplane):")
    print(f"    min={row_sums.min()} max={row_sums.max()} mean={row_sums.mean():.2f} std={row_sums.std():.2f}")

    # pairwise relation between coordinate-permutations: is any pair exact
    # reversal of one another? identity? equal?
    perms = [points[:, k] for k in range(N_COORDS)]
    rev = n - 1 - np.arange(n)
    for a, b in combinations(range(N_COORDS), 2):
        # is perms[b] == reverse-composed with perms[a] under some relabeling?
        # cheap checks: equal, exact elementwise reversal (perms[b] == n-1-perms[a] pointwise)
        if np.array_equal(perms[a], perms[b]):
            print(f"    coord {a},{b}: IDENTICAL permutations")
        if np.array_equal(perms[b], n - 1 - perms[a]):
            print(f"    coord {a},{b}: exact pointwise reversal (sigma_b = (n-1)-sigma_a)")
    # identity check
    ident = np.arange(n)
    for a in range(N_COORDS):
        if np.array_equal(perms[a], ident):
            print(f"    coord {a}: equals identity")
        if np.array_equal(perms[a], ident[::-1]):
            print(f"    coord {a}: equals reversed identity")

    print(f"  permutations (as sequences):")
    for k in range(N_COORDS):
        print(f"    sigma_{k}: {perms[k].tolist()}")


def main():
    ns = [int(x) for x in sys.argv[1:]] or [4, 5, 6, 7]
    for n in ns:
        print(f"\n=== n={n} ===")
        pts, J = search(n)
        describe(pts, n)


if __name__ == "__main__":
    main()
