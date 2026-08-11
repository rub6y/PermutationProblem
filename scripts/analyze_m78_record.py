"""Structural analysis of the new M=78 record (recurrence-depth1+basin_hop)
against the pure depth-1 recurrence seed, and a cross-record structural scan
of records.jsonl (row sums, cycle structure, PCA) for M=26/78/104/936.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rec_AI import S1_T, recurence_construction
from fw.shattering import count_shattered_triples

RECORDS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "records.jsonl")


def load_records():
    out = []
    with open(RECORDS_PATH) as f:
        for line in f:
            out.append(json.loads(line))
    return out


def best_by_M(records, M):
    cands = [r for r in records if r["M"] == M]
    cands.sort(key=lambda r: -r["J"])
    return cands[0]


def points_from_record(r):
    perms = r["perms"]
    return np.array(perms, dtype=np.int64).T  # (M, 6)


def cycle_structure(perm):
    """perm: array where perm[i] is the value at index i (permutation of range(n))."""
    n = len(perm)
    seen = [False] * n
    cycles = []
    for i in range(n):
        if seen[i]:
            continue
        cyc = []
        j = i
        while not seen[j]:
            seen[j] = True
            cyc.append(j)
            j = perm[j]
        if len(cyc) > 1:
            cycles.append(tuple(cyc))
    fixed = sum(1 for i in range(n) if perm[i] == i)
    return cycles, fixed


def main():
    records = load_records()
    r78 = best_by_M(records, 78)
    points78 = points_from_record(r78)
    M = points78.shape[0]
    print(f"=== M=78 record (opt={r78['params']['opt']}, J={r78['J']:.6f}) ===")

    row_sums = points78.sum(axis=1)
    target = 3 * (M - 1)
    print(f"row sums: all == {target}? {bool(np.all(row_sums == target))}")

    # Compare against the pure depth-1 recurrence seed (same construction,
    # no basin-hop) to see which points basin_hop actually touched.
    seed_list = recurence_construction(S1_T, 1)
    seed_points = np.array(seed_list, dtype=np.int64)

    # basin_hop_hyperplane preserves each point's *set* membership only up
    # to double-swaps; compare as sets of rows (order may differ) first,
    # then, since save_recurrence_witness.py wrote perms straight from the
    # solver's own point order (no re-sort), try direct row-order compare.
    same_order_diff = np.sum(np.any(points78 != seed_points, axis=1))
    print(f"points differing from seed in ORIGINAL row order: {same_order_diff} / {M}")

    seed_set = set(map(tuple, seed_points.tolist()))
    new_set = set(map(tuple, points78.tolist()))
    print(f"points present in both (as sets, order-independent): {len(seed_set & new_set)} / {M}")
    print(f"points only in seed (removed by search): {len(seed_set - new_set)}")
    print(f"points only in result (introduced by search): {len(new_set - seed_set)}")

    removed = seed_set - new_set
    added = new_set - seed_set
    if removed:
        print("removed points:", sorted(removed))
    if added:
        print("added points:  ", sorted(added))

    # Cycle structure of each coordinate as a permutation of [0, M-1]
    print("\ncycle structure per coordinate (excluding fixed points):")
    for k in range(6):
        col = points78[:, k]
        # col[i] = value at point i; build as permutation array indexed by point i
        cycles, fixed = cycle_structure(col.tolist())
        lens = sorted(len(c) for c in cycles)
        print(f"  coord {k}: {len(cycles)} nontrivial cycles, lengths={lens}, fixed_points={fixed}")

    # PCA / covariance structure
    print("\nPCA of point cloud (should show near-degenerate eigval along (1,...,1)):")
    X = points78.astype(np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = Xc.T @ Xc / M
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = np.ones(6) / np.sqrt(6)
    order = np.argsort(eigvals)
    for idx in order:
        v = eigvecs[:, idx]
        cos_sim = abs(float(v @ normal))
        print(f"  eigval={eigvals[idx]:.4f}  cos_sim_with_(1..1)={cos_sim:.6f}")

    # Compare against the other high-J records structurally
    print("\n=== cross-record row-sum / cycle summary ===")
    for M_target in (26, 78, 104, 936):
        r = best_by_M(records, M_target)
        pts = points_from_record(r)
        n = pts.shape[0]
        sums = pts.sum(axis=1)
        on_h = bool(np.all(sums == 3 * (n - 1)))
        J_check = count_shattered_triples(pts) / n**3
        print(f"M={M_target:4d} opt={r['params'].get('opt','?'):22s} J={r['J']:.6f} "
              f"J_recheck={J_check:.6f} on_H={on_h}")


if __name__ == "__main__":
    main()
