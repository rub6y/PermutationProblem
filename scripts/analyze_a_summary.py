"""Structural analysis of the new witnesses in ../../a_summary.txt (outside
this repo, dropped at the Math/ root) -- same analysis battery as
analyze_m78_record.py (row sums / hyperplane check, exact J recheck, cycle
structure, PCA), applied to a fresh set of witnesses spanning n=15..120 that
were not produced by any script in this repo.
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fw.shattering import count_shattered_triples

SUMMARY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "a_summary.txt",
)


def parse_a_summary(path):
    """Returns list of dicts: {n, J_claimed, mono_count, points (n,6) array}."""
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f]

    header_re = re.compile(r"n\s*=\s*(\d+)\s*(?:-->)?\s*(\d+)\s*\(([\d.]+)\)")
    records = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = header_re.search(line)
        if m:
            n = int(m.group(1))
            count = int(m.group(2))
            J = float(m.group(3))
            rows = []
            for k in range(1, 7):
                row_line = lines[i + k].strip()
                rows.append([int(x) for x in row_line.split()])
            points = np.array(rows, dtype=np.int64).T  # (n, 6)
            assert points.shape == (n, 6), (n, points.shape)
            # normalize to 0-indexed per column (n=26's block is 1-indexed
            # in the source file; every other block already starts at 0)
            points = points - points.min(axis=0, keepdims=True)
            records.append({"n": n, "count": count, "J": J, "points": points})
            i += 7
        else:
            i += 1
    return records


def cycle_structure(perm):
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
    records = parse_a_summary(SUMMARY_PATH)
    print(f"parsed {len(records)} witnesses: n = {[r['n'] for r in records]}\n")

    print(f"{'n':>5} {'J_claimed':>10} {'J_recheck':>10} {'on_H':>6} {'match':>6}")
    for r in records:
        n, pts = r["n"], r["points"]
        sums = pts.sum(axis=1)
        target = 3 * (n - 1)
        on_h = bool(np.all(sums == target))
        exact = count_shattered_triples(pts)
        J_recheck = exact / n ** 3
        match = abs(J_recheck - r["J"]) < 5e-5
        print(f"{n:5d} {r['J']:10.6f} {J_recheck:10.6f} {str(on_h):>6} {str(match):>6}"
              f"  (claimed count={r['count']}, exact count={exact})")

    print("\ncycle structure per coordinate (nontrivial cycles / fixed points):")
    for r in records:
        n, pts = r["n"], r["points"]
        summary = []
        for k in range(6):
            cycles, fixed = cycle_structure(pts[:, k].tolist())
            lens = sorted(len(c) for c in cycles)
            longest = lens[-1] if lens else 0
            summary.append(f"c{k}:nc={len(cycles)},max={longest},fix={fixed}")
        print(f"  n={n:4d}: " + "  ".join(summary))

    print("\nPCA (should show ~1 near-zero eigenvalue along (1,...,1)/sqrt(6) if on H):")
    normal = np.ones(6) / np.sqrt(6)
    for r in records:
        n, pts = r["n"], r["points"]
        X = pts.astype(np.float64)
        Xc = X - X.mean(axis=0, keepdims=True)
        cov = Xc.T @ Xc / n
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)
        smallest_idx = order[0]
        cos_sim = abs(float(eigvecs[:, smallest_idx] @ normal))
        print(f"  n={n:4d}: min eigval={eigvals[smallest_idx]:.5f}  "
              f"cos_sim_with_(1..1)={cos_sim:.6f}  all_eigvals={np.round(eigvals,3)}")

    return records


if __name__ == "__main__":
    main()
