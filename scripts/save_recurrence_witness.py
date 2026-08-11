"""Reproduce the depth-1 recurrence+basin_hop witness (deterministic, seed=0)
found by scripts/recurrence_warmstart.py and append it to records.jsonl in
the existing schema.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rec_AI import S1_T, recurence_construction
from fw.hyperplane_search import exact_J, basin_hop_hyperplane, row_sums_on_hyperplane

RECORDS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "records.jsonl")


def points_to_record(points, seed_tag, opt_name):
    """Matches the newest records.jsonl schema (e.g. 'constructed-start'
    entries): config = raw integer point coords per row, perms[k] = the
    k-th coordinate's values indexed by point (i.e. points[:, k])."""
    M = points.shape[0]
    _, on_h = row_sums_on_hyperplane(points)
    assert on_h, "witness must sit exactly on the hyperplane"
    J = exact_J(points)
    config = points.astype(np.float64).tolist()
    perms = [points[:, k].tolist() for k in range(points.shape[1])]
    return {
        "J": J,
        "M": M,
        "seed": seed_tag,
        "params": {"opt": opt_name, "note": "recurrence_construction depth-1 seed + basin_hop_hyperplane"},
        "config": config,
        "perms": perms,
    }


def main():
    depth = 1
    seed_list = recurence_construction(S1_T, depth)
    points = np.array(seed_list, dtype=np.int64)
    result = basin_hop_hyperplane(points, n_restarts=30, perturb_swaps=3, seed=0)
    print(f"reproduced best_J={result.best_J:.6f}")

    rec = points_to_record(result.best_points, seed_tag=0, opt_name="recurrence-depth1+basin_hop")
    with open(RECORDS_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"appended record: J={rec['J']}, M={rec['M']}")


if __name__ == "__main__":
    main()
