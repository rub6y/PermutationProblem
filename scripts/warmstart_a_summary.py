"""Warm-start hyperplane-preserving local search (fw.hyperplane_search.
basin_hop_hyperplane) from the best a_summary.txt witnesses, to see whether
local search can push past their J -- same idea as
recurrence_warmstart_results.md but seeded from this new, already
substantially-better-than-records.jsonl witness family instead of the
recursive construction.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.analyze_a_summary import parse_a_summary, SUMMARY_PATH
from fw.hyperplane_search import basin_hop_hyperplane, exact_J

RECORDS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "records.jsonl")


def restarts_for(n):
    # cheap at small n (O(n^2)-per-move-eval dominates cost) -- search harder
    # there since it's nearly free, same idea as steps_for() elsewhere in
    # this project (keep restarts*n^2 roughly bounded).
    return max(15, min(200, int(150_000 / (n * n))))


def main():
    records = {r["n"]: r for r in parse_a_summary(SUMMARY_PATH)}
    for n in sorted(records):
        r = records[n]
        pts = r["points"]
        seed_J = exact_J(pts)
        n_restarts = restarts_for(n)
        print(f"=== n={n}  seed J={seed_J:.6f} (a_summary claimed {r['J']:.6f})  "
              f"n_restarts={n_restarts} ===", flush=True)
        t0 = time.time()
        result = basin_hop_hyperplane(pts, n_restarts=n_restarts, perturb_swaps=3, seed=0, max_hill_climb_steps=400)
        dt = time.time() - t0
        print(f"n={n}: basin_hop best J={result.best_J:.6f}  "
              f"gain={result.best_J - seed_J:+.6f}  ({dt:.1f}s)", flush=True)
        if result.best_J > seed_J:
            out = {
                "M": n,
                "J": result.best_J,
                "perms": result.best_points.T.tolist(),
                "params": {"opt": "a_summary-warmstart+basin_hop", "seed": 0,
                           "source_n": n, "source_J": r["J"]},
            }
            with open(RECORDS_PATH, "a") as f:
                f.write(json.dumps(out) + "\n")
            print(f"  -> appended to records.jsonl (M={n}, J={result.best_J:.6f})", flush=True)


if __name__ == "__main__":
    main()
