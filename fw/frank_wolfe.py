"""The Frank-Wolfe / conditional-gradient loop of measure.tex Sec. 5:

  1. start at a measure a in A_n
  2. compute the gradient (restricted to a candidate pool, fw.objective)
  3. LP-maximize <grad, o> over o in the candidate-restricted polytope (fw.lp_oracle)
  4. line-search the exact cubic J along the segment a -> o, step there
  5. repeat, only accepting steps that exactly improve J

This module only orchestrates; the objective, the LP oracle and the
candidate-pool strategy each live in their own module (fw.objective,
fw.lp_oracle, fw.candidates) so the search strategy can be tuned without
touching the loop itself.
"""

from dataclasses import dataclass, field

import numpy as np

from fw.candidates import build_candidate_pool
from fw.lp_oracle import solve_lp_oracle
from fw.measure import mix
from fw.objective import compute_J, compute_gradient, compute_trilinear_J


@dataclass
class FrankWolfeResult:
    best_measure: object
    best_J: float
    history: list = field(default_factory=list)


def _best_step_on_segment(measure_a, measure_o):
    """Maximize the cubic f(t) = J((1-t) a + t o) over t in [0, 1] exactly,
    using the multilinear expansion of J (measure.tex: the same trick used to
    show J(mu,mu,nu) <= J(mu,mu,mu) at a maximizer)."""
    j_aaa = compute_J(measure_a)
    j_aao = compute_trilinear_J(measure_a, measure_a, measure_o)
    j_aoo = compute_trilinear_J(measure_a, measure_o, measure_o)
    j_ooo = compute_J(measure_o)

    c0 = j_aaa
    c1 = 3.0 * (j_aao - j_aaa)
    c2 = 3.0 * (j_aaa - 2.0 * j_aao + j_aoo)
    c3 = -j_aaa + 3.0 * j_aao - 3.0 * j_aoo + j_ooo

    def f(t):
        return c0 + t * (c1 + t * (c2 + t * c3))

    candidates_t = [0.0, 1.0]
    # critical points of f: c1 + 2 c2 t + 3 c3 t^2 = 0
    if abs(c3) < 1e-15:
        if abs(c2) > 1e-15:
            candidates_t.append(-c1 / (2.0 * c2))
    else:
        disc = (2.0 * c2) ** 2 - 4.0 * (3.0 * c3) * c1
        if disc >= 0:
            sqrt_disc = disc**0.5
            candidates_t.append((-2.0 * c2 + sqrt_disc) / (2.0 * 3.0 * c3))
            candidates_t.append((-2.0 * c2 - sqrt_disc) / (2.0 * 3.0 * c3))

    best_t, best_value = 0.0, c0
    for t in candidates_t:
        if 0.0 <= t <= 1.0:
            value = f(t)
            if value > best_value:
                best_t, best_value = t, value
    return best_t, best_value


def run(
    initial_measure,
    max_iterations=200,
    n_random=400,
    seed=None,
    min_improvement=1e-10,
):
    """Run Frank-Wolfe starting from `initial_measure` (must already be in
    A_n). Stops early once a step fails to improve J by at least
    `min_improvement`, since the restricted LP oracle is only a heuristic
    ascent direction, not a guaranteed one."""
    rng = np.random.default_rng(seed)
    measure = initial_measure
    current_J = compute_J(measure)
    history = [current_J]

    for iteration in range(max_iterations):
        pool = build_candidate_pool(
            measure,
            n_random=n_random,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        gradient = compute_gradient(measure, pool)
        vertex = solve_lp_oracle(pool, gradient, measure.n)

        t, candidate_J = _best_step_on_segment(measure, vertex)
        if t <= 0.0 or candidate_J <= current_J + min_improvement:
            break

        measure = mix(measure, vertex, t)
        current_J = candidate_J
        history.append(current_J)

    return FrankWolfeResult(best_measure=measure, best_J=current_J, history=history)
