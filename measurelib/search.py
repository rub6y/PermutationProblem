"""Goal 1 of KNOWLEDGE.md SS5 ("gradient descent/annealing techniques to
search for good measures"), built on measurelib's general primitives:
simulated annealing over A_n using functional.J as the energy and
shift.random_coordinate_transfer as the proposal move.

This is deliberately the simplest possible search built on the new
primitives, not a replacement for fw/hyperplane_annealing.py (which is
specialized to permutation-shaped, hyperplane-constrained moves and produced
the current best record). Its point is to exercise measurelib end to end on
a real witness and give future work (goal 2's ML-guided proposals, goal 3's
big-J construction search) a minimal, general starting point that works on
*any* measure in A_n, not just permutation-shaped ones.
"""

from dataclasses import dataclass, field

import numpy as np

from measurelib.functional import J
from measurelib.shift import random_coordinate_transfer


@dataclass
class AnnealResult:
    best_measure: object
    best_J: float
    final_measure: object
    history: list = field(default_factory=list)  # J after every accepted/rejected step


def anneal(mu, n_steps=500, epsilon_fraction=0.5, t_start=0.05, t_end=0.001, seed=None):
    """Metropolis annealing over random_coordinate_transfer moves.
    Temperature cools geometrically from t_start to t_end over n_steps; delta
    is in units of J (which lives in [0, 1]), so T is on that same scale."""
    rng = np.random.default_rng(seed)
    measure = mu
    current_J = J(measure)
    best_measure, best_J = measure, current_J
    history = [current_J]

    cooling = (t_end / t_start) ** (1.0 / max(n_steps, 1))
    temperature = t_start

    for _ in range(n_steps):
        proposal = random_coordinate_transfer(measure, rng, epsilon_fraction=epsilon_fraction)
        if proposal is not None:
            candidate, _epsilon = proposal
            candidate_J = J(candidate)
            delta = candidate_J - current_J
            if delta >= 0 or rng.random() < np.exp(delta / temperature):
                measure, current_J = candidate, candidate_J
                if current_J > best_J:
                    best_measure, best_J = measure, current_J

        history.append(current_J)
        temperature *= cooling

    return AnnealResult(best_measure=best_measure, best_J=best_J, final_measure=measure, history=history)
