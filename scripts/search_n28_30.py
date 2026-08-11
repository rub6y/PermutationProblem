"""Direction 3 from the ongoing plan: fresh from-scratch search at n=28 and
n=30 (not derived by blow-up from good_permutation.txt's n=26), to check
whether an independent witness beats the known 482/975 (~0.493628) bound.

Usage: nix-shell shell.nix --run "python3 scripts/search_n28_30.py"
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from fw.swap_search import basin_hop, exact_J

BASELINE_J = 482 / 975


def random_points(n, rng):
    pts = np.empty((n, 6), dtype=np.int64)
    for k in range(6):
        pts[:, k] = rng.permutation(n)
    return pts


def search(n, n_starts=10, n_restarts=40, seed=0):
    rng = np.random.default_rng(seed)
    best_points, best_J = None, -1.0
    perturb_schedule = [3, 6, 10]
    for s in range(n_starts):
        pts = random_points(n, rng)
        perturb = perturb_schedule[s % len(perturb_schedule)]
        t0 = time.time()
        res = basin_hop(pts, n_restarts=n_restarts, perturb_swaps=perturb,
                         seed=int(rng.integers(0, 1 << 30)))
        dt = time.time() - t0
        print(f"  n={n} start={s} perturb={perturb} best_J={res.best_J:.6f} "
              f"time={dt:.1f}s", flush=True)
        if res.best_J > best_J:
            best_J, best_points = res.best_J, res.best_points
    return best_points, best_J


def main():
    for n in (28, 30):
        print(f"=== n={n} (baseline J={BASELINE_J:.6f} from n=26) ===", flush=True)
        pts, J = search(n)
        print(f"n={n} OVERALL BEST J={J:.6f} "
              f"{'IMPROVEMENT' if J > BASELINE_J else 'no improvement'}", flush=True)
        if J > BASELINE_J:
            np.save(f"/tmp/n{n}_best.npy", pts)


if __name__ == "__main__":
    main()
