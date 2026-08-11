"""R^3 cuts of the 6-D support point clouds, saved as PNGs.

Two kinds of cut:
- coordinate cuts: raw scatter of 3 of the 6 coordinates.
- hyperplane-projection cuts: project onto the 5-D orthogonal complement of
  (1,1,1,1,1,1) (the central hyperplane's normal), then scatter 3 of those
  5 directions -- shows whether points collapse further *within* the
  hyperplane, not just relative to the cube.

Usage: nix-shell shell.nix --run "python3 scripts/plot_cuts.py"
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fw.measure import load_permutations

N_COORDS = 6
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"
GOOD_PERM_PATH = ROOT.parent / "good_permutation.txt"


def orthogonal_complement_basis():
    """An orthonormal basis of the 5-D subspace orthogonal to
    (1,1,1,1,1,1), via SVD of a matrix whose row space is that vector."""
    normal = np.ones((1, N_COORDS)) / np.sqrt(N_COORDS)
    _, _, vt = np.linalg.svd(normal, full_matrices=True)
    return vt[1:]  # rows 1..5 span the orthogonal complement


def plot_coordinate_cut(points, coords, out_path, title):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    i, j, k = coords
    ax.scatter(points[:, i], points[:, j], points[:, k], s=14, color="tab:blue")
    ax.set_xlabel(f"coord {i}")
    ax.set_ylabel(f"coord {j}")
    ax.set_zlabel(f"coord {k}")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hyperplane_projection(points, coords, out_path, title):
    basis = orthogonal_complement_basis()
    centered = points - points.mean(axis=0)
    projected = centered @ basis.T  # shape (n, 5)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    i, j, k = coords
    ax.scatter(projected[:, i], projected[:, j], projected[:, k], s=14, color="tab:green")
    ax.set_xlabel(f"hyperplane basis dir {i}")
    ax.set_ylabel(f"hyperplane basis dir {j}")
    ax.set_zlabel(f"hyperplane basis dir {k}")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# A handful of representative cuts rather than every combinations(6,3)/
# combinations(5,3) triple -- keeps output to a browsable number of PNGs
# while still showing both "adjacent" and "spread out" coordinate triples.
RAW_CUTS = [(0, 1, 2), (3, 4, 5), (0, 2, 4), (1, 3, 5)]
HYPERPLANE_CUTS = [(0, 1, 2), (2, 3, 4), (0, 2, 4)]


def plot_witness(points, label, tag):
    for coords in RAW_CUTS:
        cname = "".join(str(c) for c in coords)
        plot_coordinate_cut(
            points, coords,
            PLOTS_DIR / f"{tag}_coords_{cname}.png",
            f"{label}: raw coordinate cut {coords}",
        )
    for coords in HYPERPLANE_CUTS:
        cname = "".join(str(c) for c in coords)
        plot_hyperplane_projection(
            points, coords,
            PLOTS_DIR / f"{tag}_hyperplane_{cname}.png",
            f"{label}: hyperplane-projection cut {coords}",
        )


def pick_survey_witnesses(data):
    """Smallest-gap and a mid-gap witness from the unconstrained search, for
    comparison against good_permutation.txt."""
    unconstrained = [r for r in data if r["search"] == "unconstrained"]
    if not unconstrained:
        return []
    by_gap = sorted(unconstrained, key=lambda r: r["gap"])
    picks = [by_gap[0]]
    if len(by_gap) > 2:
        picks.append(by_gap[len(by_gap) // 2])
    return picks


def main():
    PLOTS_DIR.mkdir(exist_ok=True)

    perms = load_permutations(GOOD_PERM_PATH)
    good_points = np.array(perms, dtype=np.int64).T
    plot_witness(good_points, "good_permutation.txt (n=26)", "good_permutation")

    survey_path = RESULTS_DIR / "gap_survey.json"
    if survey_path.exists():
        data = json.loads(survey_path.read_text())
        for row in pick_survey_witnesses(data):
            pts = np.array(row["points"], dtype=np.int64)
            label = f"n={row['n']} unconstrained gap={row['gap']:.4f}"
            tag = f"survey_n{row['n']}_gap{row['gap']:.4f}"
            plot_witness(pts, label, tag)
    else:
        print(f"{survey_path} not found -- run scripts/gap_survey.py first for survey witness plots")

    print(f"Wrote plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
