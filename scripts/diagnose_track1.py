"""One-off diagnosis for the FWPush.md/plan step-3 question: why does
scripts/run_track1_scaling.py's J get worse (not better) as n grows past 26?

Compares, at matched step budget and the same seed, run_track1_scaling.py's
actual method (fw.hyperplane_annealing.multi_start_anneal, double-swaps only)
against the already-implemented but never-called multimove variant
(multi_start_anneal_multimove, double-swaps + k-cycles) to check whether the
move-set gap identified in FWPush.md is the bottleneck.

Usage: nix-shell shell.nix --run "python3 scripts/diagnose_track1.py"
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from fw.hyperplane_annealing import multi_start_anneal, multi_start_anneal_multimove
from fw.swap_search import exact_J
from scripts.gap_survey import hyperplane_seed

N_RUNS = 2
N_STEPS = 400

for n in (30,):
    rng = np.random.default_rng(n)
    seed_pts = hyperplane_seed(n, rng)

    t0 = time.time()
    res_plain = multi_start_anneal(seed_pts, n_runs=N_RUNS, n_steps=N_STEPS, seed=int(n * 1000))
    j_plain = exact_J(res_plain.best_points)
    dt_plain = time.time() - t0
    print(f"n={n}: plain J={j_plain:.6f} ({dt_plain:.1f}s)", flush=True)

    t0 = time.time()
    res_multi = multi_start_anneal_multimove(seed_pts, n_runs=N_RUNS, n_steps=N_STEPS,
                                              k_sizes=(2, 3), max_per_group=40, seed=int(n * 1000))
    j_multi = exact_J(res_multi.best_points)
    dt_multi = time.time() - t0
    print(f"n={n}: multimove(k<=3, small pool) J={j_multi:.6f} ({dt_multi:.1f}s)  "
          f"delta={j_multi - j_plain:+.6f}", flush=True)
