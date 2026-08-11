"""Survey of near-optimal 6-tuples across several n, both unconstrained and
hyperplane-constrained, to build the data needed to test whether the gap
`0.5 - J` correlates with distance from the central hyperplane
`sum_k sigma_k(i) = 3(n-1)`.

Usage: nix-shell shell.nix --run "python3 scripts/gap_survey.py"
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from fw.hyperplane_search import basin_hop_hyperplane
from fw.measure import load_permutations
from fw.swap_search import basin_hop, exact_J

N_COORDS = 6
NS = [6, 8, 10, 14, 18, 22, 26]
N_STARTS = 4
N_RESTARTS = 15
PERTURB_SWAPS = 3
GOOD_PERM_PATH = Path(__file__).resolve().parent.parent.parent / "good_permutation.txt"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def hyperplane_seed(n, rng):
    """A random point cloud exactly on the central hyperplane
    `sum_k x_k(i) = 3(n-1)`. Pair coordinates (0,1), (2,3), (4,5); within
    each pair use `pi(i)` and `n-1-pi(i)` for an independently random
    permutation `pi`, so each pair always contributes exactly `n-1` to
    every row's sum regardless of `pi`."""
    cols = np.empty((n, N_COORDS), dtype=np.int64)
    for pair in range(N_COORDS // 2):
        pi = rng.permutation(n)
        cols[:, 2 * pair] = pi
        cols[:, 2 * pair + 1] = n - 1 - pi
    return cols


def row_sum_deviation(points, n):
    target = N_COORDS * (n - 1) / 2  # mean per-coordinate value (n-1)/2, times 6 coords
    row_sums = points.sum(axis=1)
    return float(np.mean(np.abs(row_sums - target))) / (n - 1)


def collect_unconstrained(n, rng, results):
    for s in range(N_STARTS):
        pts = np.stack([rng.permutation(n) for _ in range(N_COORDS)], axis=1)
        res = basin_hop(pts, n_restarts=N_RESTARTS, perturb_swaps=PERTURB_SWAPS,
                         seed=int(rng.integers(0, 1 << 30)))
        results.append({
            "n": n, "search": "unconstrained", "start": s,
            "J": res.best_J, "gap": 0.5 - res.best_J,
            "deviation": row_sum_deviation(res.best_points, n),
            "points": res.best_points.tolist(),
        })


def collect_hyperplane(n, rng, results):
    for s in range(N_STARTS):
        pts = hyperplane_seed(n, rng)
        res = basin_hop_hyperplane(pts, n_restarts=N_RESTARTS, perturb_swaps=PERTURB_SWAPS,
                                    seed=int(rng.integers(0, 1 << 30)))
        results.append({
            "n": n, "search": "hyperplane", "start": s,
            "J": res.best_J, "gap": 0.5 - res.best_J,
            "deviation": row_sum_deviation(res.best_points, n),
            "points": res.best_points.tolist(),
        })


def collect_good_permutation(results):
    perms = load_permutations(GOOD_PERM_PATH)
    points = np.array(perms, dtype=np.int64).T
    n = points.shape[0]
    J = exact_J(points)
    results.append({
        "n": n, "search": "good_permutation.txt", "start": 0,
        "J": J, "gap": 0.5 - J,
        "deviation": row_sum_deviation(points, n),
        "points": points.tolist(),
    })


def main():
    rng = np.random.default_rng(0)
    results = []

    collect_good_permutation(results)

    for n in NS:
        print(f"=== n={n} ===")
        collect_unconstrained(n, rng, results)
        collect_hyperplane(n, rng, results)
        for r in results:
            if r["n"] == n:
                print(f"  {r['search']:>22s} start={r['start']} J={r['J']:.6f} "
                      f"gap={r['gap']:.6f} deviation={r['deviation']:.6f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "gap_survey.json"
    out_path.write_text(json.dumps(results))
    print(f"\nWrote {len(results)} witnesses to {out_path}")


if __name__ == "__main__":
    main()
