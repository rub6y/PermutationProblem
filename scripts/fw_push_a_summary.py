"""Push a_summary.txt witnesses further with the real Frank-Wolfe / gradient
loop (fw.frank_wolfe.run), instead of the discrete double-swap basin-hop
already tried in warmstart_a_summary.py. FW works in the general marginal-
uniform measure space A_n (fw.measure), not just permutation point clouds,
so it can explore directions (fractional mass reweighting, single-coordinate
moves) that basin_hop_hyperplane's paired double-swaps cannot reach --
useful now that we know (via PCA, analyze_a_summary.py) that near-optimal
measures concentrate almost exactly on the central hyperplane H, which the
candidate pool's random/neighbor points mostly do NOT lie on.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.analyze_a_summary import parse_a_summary, SUMMARY_PATH
from fw.measure import Measure
from fw.objective import compute_J, compute_gradient
from fw.lp_oracle import solve_lp_oracle
from fw.frank_wolfe import _best_step_on_segment
from fw.candidates import full_neighbor_points
from fw.measure import mix

TARGET_NS = [60, 100, 120]


def hyperplane_random_points(n, count, rng):
    """Sample points with 5 coords uniform in [0,n) and the 6th solved to
    land exactly on H = {sum x_k = 3(n-1)}, retrying on out-of-range draws --
    unlike fw.candidates.random_points, which samples the full grid and (per
    analyze_a_summary.py's PCA check) lands almost entirely off H."""
    target = 3 * (n - 1)
    out = []
    while len(out) < count:
        five = rng.integers(0, n, size=5)
        last = target - int(five.sum())
        if 0 <= last < n:
            out.append(np.append(five, last))
    return np.array(out, dtype=np.int64)


def build_h_biased_pool(measure, n_random, rng):
    parts = [
        measure.points,
        hyperplane_random_points(measure.n, n_random, rng),
        full_neighbor_points(measure.points, measure.n),
    ]
    return np.unique(np.concatenate(parts, axis=0), axis=0)


def fw_run_h_biased(measure, max_iterations, n_random, seed):
    rng = np.random.default_rng(seed)
    current_J = compute_J(measure)
    history = [current_J]
    for _ in range(max_iterations):
        pool = build_h_biased_pool(measure, n_random, rng)
        gradient = compute_gradient(measure, pool)
        vertex = solve_lp_oracle(pool, gradient, measure.n)
        t, candidate_J = _best_step_on_segment(measure, vertex)
        if t <= 0.0 or candidate_J <= current_J + 1e-10:
            break
        measure = mix(measure, vertex, t)
        current_J = candidate_J
        history.append(current_J)
    return measure, current_J, history


def main():
    records = {r["n"]: r for r in parse_a_summary(SUMMARY_PATH)}
    for n in TARGET_NS:
        r = records[n]
        pts = r["points"]
        weights = np.full(n, 1.0 / n)
        measure = Measure(points=pts, weights=weights, n=n)
        seed_J = compute_J(measure)
        print(f"=== n={n}  seed J={seed_J:.6f} (H-biased candidate pool) ===", flush=True)
        t0 = time.time()
        best_measure, best_J, history = fw_run_h_biased(measure, max_iterations=60, n_random=600, seed=0)
        dt = time.time() - t0
        print(f"n={n}: FW(H-biased) best J={best_J:.6f}  gain={best_J - seed_J:+.6f}  "
              f"iters={len(history)-1}  support={best_measure.support_size}  ({dt:.1f}s)", flush=True)
        print(f"  history: {[round(h,6) for h in history]}", flush=True)


if __name__ == "__main__":
    main()
