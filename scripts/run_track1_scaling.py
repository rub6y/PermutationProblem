"""Push n beyond 26 using Track 1's approach (SA restricted to the
hyperplane-preserving move class), using the geometric observation itself as
the seeding mechanism: fresh point clouds start exactly on the central
hyperplane sum_k sigma_k(i) = 3(n-1) via hyperplane_seed's reversal-pairing
construction, then SA searches only within that hyperplane. Tracks how the
best-found J trends with n -- the conjecture is lim J = 1/2 as n -> infinity,
so this checks whether hyperplane-constrained search approaches that limit
faster/higher than unconstrained search did in gap_survey.json.

Usage: nix-shell shell.nix --run "python3 scripts/run_track1_scaling.py"
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fw.hyperplane_annealing import multi_start_anneal
from fw.swap_search import exact_J
from scripts.gap_survey import hyperplane_seed

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"

NS = [30, 34, 40, 50, 60]
N_RUNS = 3
N_STEPS = 1000


def save_witness(points, path):
    perms = points.T.tolist()
    lines = ["S1 = ["]
    for p in perms:
        lines.append(f"  {tuple(p)},")
    lines.append("]")
    path.write_text("\n".join(lines) + "\n")


def main():
    results = []
    for n in NS:
        print(f"=== n={n} ===")
        t0 = time.time()
        rng = np.random.default_rng(n)  # distinct, reproducible seed per n
        seed_pts = hyperplane_seed(n, rng)
        res = multi_start_anneal(seed_pts, n_runs=N_RUNS, n_steps=N_STEPS, seed=int(n * 1000))
        verified_J = exact_J(res.best_points)
        dt = time.time() - t0
        print(f"  best J = {verified_J:.6f} (self-reported {res.best_J:.6f}), "
              f"gap to 0.5 = {0.5 - verified_J:.6f}, took {dt:.1f}s")
        results.append({"n": n, "J": verified_J, "gap": 0.5 - verified_J, "seconds": dt})
        save_witness(res.best_points, RESULTS_DIR / f"annealing_best_n{n}.txt")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "track1_scaling.json").write_text(json.dumps(results))

    # include the n=26 point for context
    n26_path = RESULTS_DIR / "annealing_best_n26.txt"
    if n26_path.exists():
        from fw.measure import load_permutations
        perms = load_permutations(n26_path)
        pts26 = np.array(perms, dtype=np.int64).T
        results = [{"n": 26, "J": exact_J(pts26), "gap": 0.5 - exact_J(pts26), "seconds": None}] + results

    ns_plot = [r["n"] for r in results]
    Js_plot = [r["J"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns_plot, Js_plot, "o-", label="Track 1 SA best J (hyperplane-seeded)")
    ax.axhline(0.5, color="red", linestyle="--", label="conjectured limit 0.5")
    ax.set_xlabel("n")
    ax.set_ylabel("J")
    ax.set_title("Track 1 SA best-found J vs n")
    ax.legend()
    fig.tight_layout()
    PLOTS_DIR.mkdir(exist_ok=True)
    fig.savefig(PLOTS_DIR / "track1_scaling_J_vs_n.png", dpi=150)
    plt.close(fig)

    print(f"\nWrote {RESULTS_DIR / 'track1_scaling.json'} and plots/track1_scaling_J_vs_n.png")


if __name__ == "__main__":
    main()
