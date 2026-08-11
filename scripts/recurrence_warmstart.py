"""Warm-start hyperplane-preserving local search from a rec_AI recursion seed
(step 4 of NEXT_recurrence_scaling_plan.md), instead of Track 1's random
hyperplane_seed -- diagnosed root cause of Track 1's regression was cold
starting near J~0 at every n.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rec_AI import S1_T, recurence_construction
from fw.hyperplane_search import exact_J, basin_hop_hyperplane, row_sums_on_hyperplane

BEST_RECORD_J = 0.49431042330450614


def main():
    for depth in (1, 2):
        seed_list = recurence_construction(S1_T, depth)
        points = np.array(seed_list, dtype=np.int64)
        M = points.shape[0]
        seed_J = exact_J(points)
        print(f"depth={depth} M={M} seed_J={seed_J:.6f}")

        t0 = time.time()
        result = basin_hop_hyperplane(points, n_restarts=30, perturb_swaps=3, seed=0)
        dt = time.time() - t0
        _, on_h = row_sums_on_hyperplane(result.best_points)
        print(f"  basin_hop: best_J={result.best_J:.6f} on_H={on_h} ({dt:.1f}s, "
              f"{'BEATS' if result.best_J > BEST_RECORD_J else 'below'} record {BEST_RECORD_J:.6f})")


if __name__ == "__main__":
    main()
