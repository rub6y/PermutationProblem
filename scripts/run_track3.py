"""Track 3: simulated annealing with a mixed move pool of 2-point double
swaps and k-cycles (k=3,4) on the hyperplane, trying to beat the current
best-known witness (results/annealing_best_n26.txt, J=8682/17576~=0.493969,
found by Track 1's 2-point-only SA).

Usage: nix-shell shell.nix --run "python3 scripts/run_track3.py"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fw.hyperplane_annealing import multi_start_anneal_multimove
from fw.measure import load_permutations
from fw.swap_search import exact_J
from scripts.gap_survey import hyperplane_seed

ROOT = Path(__file__).resolve().parent.parent
GOOD_PERM_PATH = ROOT.parent / "good_permutation.txt"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"

BASELINE_J = 8676 / 17576
N_RUNS = 3
N_STEPS = 800
K_SIZES = (2, 3, 4)
MAX_PER_GROUP = 50


def save_witness(points, path):
    perms = points.T.tolist()
    lines = ["S1 = ["]
    for p in perms:
        lines.append(f"  {tuple(p)},")
    lines.append("]")
    path.write_text("\n".join(lines) + "\n")


def current_best():
    """Best verified witness on disk: annealing_best_n26.txt if present and
    beats the original baseline, else good_permutation.txt."""
    best_path = RESULTS_DIR / "annealing_best_n26.txt"
    if best_path.exists():
        perms = load_permutations(best_path)
        points = np.array(perms, dtype=np.int64).T
        J = exact_J(points)
        if J > BASELINE_J:
            return points, J
    perms = load_permutations(GOOD_PERM_PATH)
    points = np.array(perms, dtype=np.int64).T
    return points, exact_J(points)


def main():
    good_points, best_J_on_disk = current_best()
    assert abs(exact_J(good_points) - best_J_on_disk) < 1e-9, "sanity check failed"
    print(f"current best J on disk = {best_J_on_disk:.6f} (baseline was {BASELINE_J:.6f})")

    print(f"\n=== Track 3 SA (k_sizes={K_SIZES}) from current best on disk ===")
    result_from_best = multi_start_anneal_multimove(
        good_points, n_runs=N_RUNS, n_steps=N_STEPS, seed=100,
        k_sizes=K_SIZES, max_per_group=MAX_PER_GROUP)
    print(f"best J = {result_from_best.best_J:.6f}")

    print(f"\n=== Track 3 SA (k_sizes={K_SIZES}) from fresh hyperplane-seeded random start (n=26) ===")
    rng = np.random.default_rng(2)
    fresh = hyperplane_seed(26, rng)
    result_from_fresh = multi_start_anneal_multimove(
        fresh, n_runs=N_RUNS, n_steps=N_STEPS, seed=101,
        k_sizes=K_SIZES, max_per_group=MAX_PER_GROUP)
    print(f"best J = {result_from_fresh.best_J:.6f}")

    best_overall = max([result_from_best, result_from_fresh], key=lambda r: r.best_J)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(result_from_best.history, label="from current best on disk", alpha=0.8)
    ax.plot(result_from_fresh.history, label="from fresh hyperplane seed", alpha=0.8)
    ax.axhline(best_J_on_disk, color="red", linestyle="--", label=f"current best {best_J_on_disk:.6f}")
    ax.set_xlabel("SA step (best run of multi_start_anneal_multimove)")
    ax.set_ylabel("J (current, not best-so-far)")
    ax.set_title("Track 3: SA with 2/3/4-point hyperplane moves (n=26)")
    ax.legend()
    fig.tight_layout()
    PLOTS_DIR.mkdir(exist_ok=True)
    fig.savefig(PLOTS_DIR / "annealing_curve_track3_n26.png", dpi=150)
    plt.close(fig)

    if best_overall.best_J > best_J_on_disk + 1e-9:
        verified = exact_J(best_overall.best_points)
        print(f"\nIMPROVEMENT FOUND: J={verified:.6f} > previous best on disk {best_J_on_disk:.6f}")
        RESULTS_DIR.mkdir(exist_ok=True)
        save_witness(best_overall.best_points, RESULTS_DIR / "annealing_best_n26.txt")
        print(f"Saved to {RESULTS_DIR / 'annealing_best_n26.txt'}")
    else:
        print(f"\nNo improvement: best J across all Track 3 runs = {best_overall.best_J:.6f} "
              f"vs current best on disk {best_J_on_disk:.6f}")


if __name__ == "__main__":
    main()
