"""Track 2: search for closed-form (algebraic) permutation families that land
exactly on the central hyperplane by construction, to see if any beats the
search-based optima -- analogous to how the pair-shattering problem's proven
maximizer is a closed form (the antidiagonal), not a search result.

Family: for each of the 3 coordinate pairs (0,1),(2,3),(4,5), use
pi_pair(i) = (a*i + b) mod n for coprime a, and set the pair's two
coordinates to pi_pair(i) and n-1-pi_pair(i) (guarantees hyperplane
membership exactly, generalizing scripts/gap_survey.hyperplane_seed's random
version to a structured linear family). Sweep (a, b) per pair for small n
exhaustively, and compare best found against fw.swap_search basin-hop optima
already recorded in results/gap_survey.json.

Usage: nix-shell shell.nix --run "python3 scripts/algebraic_construction_search.py"
"""

import json
import math
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from fw.swap_search import exact_J

N_COORDS = 6
N_PAIRS = N_COORDS // 2
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
GOOD_PERM_PATH = ROOT.parent / "good_permutation.txt"


def linear_permutation(n, a, b):
    i = np.arange(n)
    return (a * i + b) % n


def build_points(n, params):
    """params: list of N_PAIRS (a, b) tuples."""
    cols = np.empty((n, N_COORDS), dtype=np.int64)
    for pair, (a, b) in enumerate(params):
        pi = linear_permutation(n, a, b)
        cols[:, 2 * pair] = pi
        cols[:, 2 * pair + 1] = n - 1 - pi
    return cols


def coprime_units(n):
    return [a for a in range(1, n) if math.gcd(a, n) == 1]


def sweep_linear_family(n, max_configs=3000):
    """Exhaustively sweep (a, b) per pair when the search space is small
    enough, else random-sample max_configs configurations."""
    units = coprime_units(n)
    b_range = range(n)
    per_pair_choices = [(a, b) for a in units for b in b_range]
    total = len(per_pair_choices) ** N_PAIRS

    best_J, best_params = -1.0, None
    rng = np.random.default_rng(0)

    if total <= max_configs:
        combos = product(per_pair_choices, repeat=N_PAIRS)
    else:
        idx = rng.integers(0, len(per_pair_choices), size=(max_configs, N_PAIRS))
        combos = (tuple(per_pair_choices[idx[row, p]] for p in range(N_PAIRS))
                  for row in range(max_configs))

    n_tried = 0
    for params in combos:
        pts = build_points(n, params)
        # skip degenerate configs where two points coincide (a,b not
        # actually giving 6 distinct coordinates per point isn't possible
        # here since each column is itself a permutation, but duplicate rows
        # across the whole 6-tuple are impossible too -- still guard cheaply)
        J = exact_J(pts)
        n_tried += 1
        if J > best_J:
            best_J, best_params = J, params

    return best_J, best_params, n_tried, min(total, max_configs)


def main():
    survey_path = RESULTS_DIR / "gap_survey.json"
    search_best = {}
    if survey_path.exists():
        for row in json.loads(survey_path.read_text()):
            n = row["n"]
            if row["J"] > search_best.get(n, -1.0):
                search_best[n] = row["J"]

    print(f"{'n':>4} {'algebraic best J':>18} {'search best J':>16} {'beats search?':>14} configs tried")
    results = []
    for n in [6, 8, 10, 14, 18]:
        best_J, best_params, n_tried, total = sweep_linear_family(n)
        sbest = search_best.get(n)
        beats = (sbest is not None) and (best_J > sbest + 1e-9)
        print(f"{n:>4} {best_J:>18.6f} {sbest if sbest is not None else float('nan'):>16.6f} "
              f"{str(beats):>14} {n_tried}/{total}")
        results.append({"n": n, "algebraic_best_J": best_J, "params": best_params,
                         "search_best_J": sbest, "beats_search": beats})

    # Also test the family at n=26 directly against the annealing-improved
    # witness, since that's the number that actually matters for the
    # conjecture's current best-known value.
    n = 26
    best_J, best_params, n_tried, total = sweep_linear_family(n, max_configs=3000)
    annealed_best = 8682 / 17576
    print(f"\nn=26 algebraic best J={best_J:.6f} (params={best_params}) "
          f"vs current best known J={annealed_best:.6f}: "
          f"{'BEATS IT' if best_J > annealed_best + 1e-9 else 'does not beat it'}")
    results.append({"n": n, "algebraic_best_J": best_J, "params": best_params,
                     "search_best_J": annealed_best, "beats_search": best_J > annealed_best + 1e-9})

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "algebraic_construction_search.json").write_text(json.dumps(results, default=str))
    print(f"\nWrote results to {RESULTS_DIR / 'algebraic_construction_search.json'}")


if __name__ == "__main__":
    main()
