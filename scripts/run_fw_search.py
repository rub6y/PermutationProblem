#!/usr/bin/env python3
"""CLI entry point: run the Frank-Wolfe search of fw.frank_wolfe starting
from good_permutation.txt, and report whether it improves on the known
~0.494 lower bound."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fw.frank_wolfe import run
from fw.measure import check_marginals, load_permutations_as_measure
from fw.objective import compute_J

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT.parent / "good_permutation.txt"
BASELINE_J = 482 / 975  # known lower bound from good_permutation.txt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--n-random", type=int, default=400)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    initial_measure = load_permutations_as_measure(args.input)
    check_marginals(initial_measure)
    baseline_J = compute_J(initial_measure)
    print(f"Loaded {args.input}: n={initial_measure.n}, J={baseline_J:.6f}")

    result = run(
        initial_measure,
        max_iterations=args.max_iterations,
        n_random=args.n_random,
        seed=args.seed,
    )

    print(f"Frank-Wolfe steps taken: {len(result.history) - 1}")
    print(f"J history: {[round(j, 6) for j in result.history]}")
    print(f"Best J found: {result.best_J:.6f} (support size {result.best_measure.support_size})")
    print(f"Reference baseline (good_permutation.txt / 482/975): {BASELINE_J:.6f}")
    if result.best_J > BASELINE_J:
        print("IMPROVEMENT over the known 0.494 lower bound.")
    else:
        print("No improvement over the known lower bound with this run's parameters.")


if __name__ == "__main__":
    main()
