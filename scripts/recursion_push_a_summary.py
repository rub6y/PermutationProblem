"""Push past n=120 using rec_AI.py's base-3 digit-expansion recursion
(recurrence_warmstart_results.md's proven approach: depth-1 recursion +
basin_hop beat the old record) seeded from the *best a_summary-derived*
witness at each n, instead of good_permutation.txt's n=26 seed.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rec_AI import recurence_construction, Sym_T
from fw.shattering import count_shattered_triples
from fw.hyperplane_search import basin_hop_hyperplane, exact_J

RECORDS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "records.jsonl")


def load_records():
    with open(RECORDS_PATH) as f:
        return [json.loads(line) for line in f]


def best_by_M(records, M):
    cands = [r for r in records if r["M"] == M]
    cands.sort(key=lambda r: -r["J"])
    return cands[0]


def main():
    records = load_records()
    seed_M = 120
    r = best_by_M(records, seed_M)
    print(f"seed: M={seed_M}, opt={r['params'].get('opt')}, J={r['J']:.6f}", flush=True)
    pts = np.array(r["perms"], dtype=np.int64).T  # (M, 6)
    seed_tensor = [tuple(row) for row in pts.tolist()]

    t0 = time.time()
    expanded = recurence_construction(seed_tensor, 1, digits=Sym_T[:3])
    expanded_pts = np.array(expanded, dtype=np.int64)
    M2 = expanded_pts.shape[0]
    exact_count = count_shattered_triples(expanded_pts)
    J2 = exact_count / M2 ** 3
    sums = expanded_pts.sum(axis=1)
    on_h = bool(np.all(sums == sums[0]))
    print(f"depth-1 expansion: M={M2}  J={J2:.6f}  on_H(constant row sum)={on_h}  "
          f"row_sum={sums[0]}  ({time.time()-t0:.1f}s to build+score)", flush=True)

    print(f"\nrunning basin_hop_hyperplane warm-start on M={M2} (this is the slow part)...", flush=True)
    t0 = time.time()
    result = basin_hop_hyperplane(expanded_pts, n_restarts=5, perturb_swaps=3, seed=0, max_hill_climb_steps=150)
    dt = time.time() - t0
    print(f"M={M2}: basin_hop best J={result.best_J:.6f}  gain_over_expansion={result.best_J - J2:+.6f}  ({dt:.1f}s)",
          flush=True)

    if result.best_J > 0:
        out = {
            "M": M2,
            "J": result.best_J,
            "perms": result.best_points.T.tolist(),
            "params": {"opt": "a_summary-n120-recursion_depth1+basin_hop", "seed": 0,
                       "source_M": seed_M, "source_J": r["J"]},
        }
        with open(RECORDS_PATH, "a") as f:
            f.write(json.dumps(out) + "\n")
        print(f"  -> appended to records.jsonl (M={M2}, J={result.best_J:.6f})", flush=True)


if __name__ == "__main__":
    main()
