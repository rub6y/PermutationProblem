"""Falsification test from 02_perturbative_expansion.md, Section 2.

Loads the current best record (records.jsonl, J=0.495195, M=78) and asks:
does the exact LP oracle (fw/lp_oracle.py), run against the exact gradient
of J at this measure, find ANY point (within the candidate neighborhood)
strictly improving on <grad, mu*> = 3*J(mu*)? If not, mu* is a first-order
stationary point of J restricted to A_n within that neighborhood -- further
gains at this resolution require a genuinely different search move, not
more of the same kind of local perturbation.
"""

import json
import sys
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fw.measure import Measure, measure_from_permutations  # noqa: E402
from fw.objective import compute_J, compute_gradient  # noqa: E402
from fw.candidates import build_candidate_pool, full_neighbor_points  # noqa: E402
from fw.lp_oracle import solve_lp_oracle  # noqa: E402


def load_record(target_j):
    with open(ROOT.parent / "records.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            if abs(rec["J"] - target_j) < 1e-9:
                return rec
    raise ValueError("record not found")


def main():
    rec = load_record(0.49519546856824964)
    perms = rec["perms"]
    mu = measure_from_permutations(perms)
    n = mu.n
    J0 = compute_J(mu)
    print(f"loaded record: M={mu.support_size}, n={n}, J={J0:.8f} (file says {rec['J']:.8f})")

    # Full single-coordinate neighborhood of every support point, plus a
    # generous random sample, as the LP oracle's candidate pool -- this is
    # the same pool machinery Frank-Wolfe itself uses, not a new mechanism.
    pool = build_candidate_pool(mu, n_random=20000, seed=0)
    print(f"candidate pool size: {pool.shape[0]}")

    grad = compute_gradient(mu, pool)
    baseline = 3.0 * J0  # <grad, mu> = 3 J(mu) since grad_p = 3 sum K(p,q,r) w_q w_r
    print(f"<grad, mu*> (via def, 3J) = {baseline:.8f}")

    oracle = solve_lp_oracle(pool, grad, n)
    # <grad, oracle>: gather grad values at oracle.points positions within pool
    point_to_grad = {tuple(p): g for p, g in zip(pool, grad)}
    oracle_value = sum(point_to_grad[tuple(p)] * w for p, w in zip(oracle.points, oracle.weights))
    print(f"<grad, oracle> (LP optimum over pool) = {oracle_value:.8f}")
    print(f"ascent slack = <grad,oracle> - <grad,mu*> = {oracle_value - baseline:.8e}")

    if oracle_value - baseline > 1e-6:
        print("=> ASCENT DIRECTION FOUND: mu* is NOT LP-stationary w.r.t. this pool.")
        # One full Frank-Wolfe line search step along mu* -> oracle:
        from fw.objective import compute_trilinear_J
        best_t, best_J = 0.0, J0
        for t in np.linspace(0.0, 1.0, 201):
            from fw.measure import mix
            mixed = mix(mu, oracle, t)
            Jt = compute_J(mixed)
            if Jt > best_J:
                best_J, best_t = Jt, t
        print(f"line search: best t={best_t:.4f}, J={best_J:.8f} (gain {best_J - J0:.8f})")
    else:
        print("=> NO ascent direction found in this pool: mu* is a first-order")
        print("   stationary point of J restricted to A_n w.r.t. single-coordinate")
        print("   moves + random resampling. Gains at fixed n likely exhausted.")


if __name__ == "__main__":
    main()
