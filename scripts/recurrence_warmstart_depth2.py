"""Depth-2 (M=234) warm-start with a smaller basin-hop budget than the
original full run (which was killed after ~40 CPU-min at n_restarts=30).
Times the first hill-climb alone to calibrate, then runs a reduced
basin-hop budget.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rec_AI import S1_T, recurence_construction
from fw.hyperplane_search import (
    exact_J,
    hill_climb_hyperplane,
    basin_hop_hyperplane,
    row_sums_on_hyperplane,
)

BEST_RECORD_J = 0.49519546856824964  # current best, from depth-1 warm-start
N_RESTARTS = 4
MAX_HILL_CLIMB_STEPS = 150  # reduced from 1000: M=234's candidate generation
                             # (_candidate_moves, pure-Python) is the real
                             # cost driver per step, not just numba compute


def main():
    seed_list = recurence_construction(S1_T, 2)
    points = np.array(seed_list, dtype=np.int64)
    M = points.shape[0]
    seed_J = exact_J(points)
    print(f"depth=2 M={M} seed_J={seed_J:.6f}", flush=True)

    t0 = time.time()
    _, single_J, steps = hill_climb_hyperplane(points, max_steps=MAX_HILL_CLIMB_STEPS)
    dt = time.time() - t0
    print(f"single hill_climb (max_steps={MAX_HILL_CLIMB_STEPS}): J={single_J:.6f} "
          f"steps_taken={steps} ({dt:.1f}s, {dt/max(steps,1):.2f}s/step)", flush=True)

    t0 = time.time()
    result = basin_hop_hyperplane(
        points, n_restarts=N_RESTARTS, perturb_swaps=3, seed=0,
        max_hill_climb_steps=MAX_HILL_CLIMB_STEPS,
    )
    dt = time.time() - t0
    _, on_h = row_sums_on_hyperplane(result.best_points)
    print(
        f"basin_hop (n_restarts={N_RESTARTS}, max_steps={MAX_HILL_CLIMB_STEPS}): "
        f"best_J={result.best_J:.6f} on_H={on_h} "
        f"({dt:.1f}s, {'BEATS' if result.best_J > BEST_RECORD_J else 'below'} "
        f"record {BEST_RECORD_J:.6f})",
        flush=True,
    )
    print(f"history={result.history}", flush=True)


if __name__ == "__main__":
    main()
