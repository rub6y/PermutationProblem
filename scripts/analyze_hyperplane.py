"""Analyze scripts/gap_survey.py's output: does the gap `0.5 - J` correlate
with distance from the central hyperplane `sum_k x_k = 3(n-1)`, and do
near-optimal point clouds concentrate near a codimension-1 subspace (PCA
eigenvalue check) with normal close to `(1,1,1,1,1,1)`?

Usage: nix-shell shell.nix --run "python3 scripts/analyze_hyperplane.py"
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / ".." / "results"
RESULTS_DIR = RESULTS_DIR.resolve()
PLOTS_DIR = Path(__file__).resolve().parent / ".." / "plots"
PLOTS_DIR = PLOTS_DIR.resolve()

N_COORDS = 6
ONES = np.ones(N_COORDS) / np.sqrt(N_COORDS)


def pca_eigen(points):
    """Eigenvalues (ascending) and eigenvectors of the covariance matrix of
    `points`, plus the cosine similarity of the smallest-eigenvalue
    eigenvector with the all-ones direction (the central hyperplane's
    normal)."""
    centered = points - points.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    smallest_vec = eigvecs[:, 0]
    cos_sim = abs(float(np.dot(smallest_vec, ONES)))
    return eigvals, cos_sim


def main():
    data = json.loads((RESULTS_DIR / "gap_survey.json").read_text())
    for row in data:
        points = np.array(row["points"], dtype=np.float64)
        eigvals, cos_sim = pca_eigen(points)
        row["eigvals"] = eigvals.tolist()
        row["smallest_eig_ratio"] = float(eigvals[0] / eigvals[-1]) if eigvals[-1] > 0 else float("nan")
        row["normal_cos_sim"] = cos_sim

    print(f"{'n':>4} {'search':>22} {'J':>9} {'gap':>9} {'row-sum dev':>12} "
          f"{'eig[0]/eig[5]':>14} {'cos(normal)':>12}")
    for row in sorted(data, key=lambda r: (r["n"], r["search"])):
        print(f"{row['n']:>4} {row['search']:>22} {row['J']:>9.6f} {row['gap']:>9.6f} "
              f"{row['deviation']:>12.6f} {row['smallest_eig_ratio']:>14.6f} "
              f"{row['normal_cos_sim']:>12.6f}")

    PLOTS_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    for search, marker, color in [
        ("unconstrained", "o", "tab:orange"),
        ("hyperplane", "s", "tab:blue"),
        ("good_permutation.txt", "*", "tab:red"),
    ]:
        rows = [r for r in data if r["search"] == search]
        if not rows:
            continue
        gaps = [r["gap"] for r in rows]
        devs = [r["deviation"] for r in rows]
        ax.scatter(devs, gaps, marker=marker, color=color, label=search, s=60 if search.startswith("good") else 30)
    ax.set_xlabel("row-sum deviation from central hyperplane (normalized)")
    ax.set_ylabel("gap = 0.5 - J")
    ax.set_title("Gap to conjectured limit vs. distance from central hyperplane")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "gap_vs_deviation.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for search, marker, color in [
        ("unconstrained", "o", "tab:orange"),
        ("hyperplane", "s", "tab:blue"),
        ("good_permutation.txt", "*", "tab:red"),
    ]:
        rows = [r for r in data if r["search"] == search]
        if not rows:
            continue
        gaps = [r["gap"] for r in rows]
        ratios = [r["smallest_eig_ratio"] for r in rows]
        ax.scatter(gaps, ratios, marker=marker, color=color, label=search, s=60 if search.startswith("good") else 30)
    ax.set_xlabel("gap = 0.5 - J")
    ax.set_ylabel("smallest / largest covariance eigenvalue")
    ax.set_yscale("log")
    ax.set_title("Codimension-1 concentration vs. gap")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "eigenvalue_ratio_vs_gap.png", dpi=150)
    plt.close(fig)

    out_path = RESULTS_DIR / "gap_survey_analyzed.json"
    out_path.write_text(json.dumps(data))
    print(f"\nWrote analysis to {out_path} and plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
