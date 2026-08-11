"""Map J vs. depth for rec_AI.recurence_construction, and record row-sum /
digit-structure sanity checks (steps 2-3 of NEXT_recurrence_scaling_plan.md).
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rec_AI import S1_T, Sym_T, recurence_construction, compute_J_fast


def row_sums_ok(tensor_list, n):
    target = 3 * (n - 1)
    return all(sum(row) == target for row in tensor_list)


def main():
    print(f"Sym_T = {Sym_T} (len={len(Sym_T)})")
    print(f"digit sums = {[sum(d) for d in Sym_T]}")

    for depth in range(0, 5):
        t0 = time.time()
        tensor_list = recurence_construction(S1_T, depth)
        M = len(tensor_list)
        n = M  # point cloud embeds into [0, M-1]^6 conceptually via /(M-1)
        ok = row_sums_ok(tensor_list, M)
        tensor_np = np.array(tensor_list, dtype=np.int64)
        J = compute_J_fast(tensor_np)
        dt = time.time() - t0
        print(f"depth={depth} M={M} on_H={ok} J={J:.6f} ({dt:.1f}s)")


if __name__ == "__main__":
    main()
