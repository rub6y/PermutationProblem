"""Scaffolding for proof-side work on lim J(mu_n): wraps a resolution-indexed
family of measures (e.g. a recursive blow-up like rec_AI.py's digit
expansion, or a future growing-gadget d-sweep) together with its J values,
and exposes trend/plateau diagnostics -- generalizing the ad hoc convergence
checks currently redone per-script (e.g. recurrence_convergence.py) so any
new sequence construction gets them for free.
"""

from dataclasses import dataclass, field

from measurelib.functional import J


@dataclass
class MeasureSequence:
    """mu_n for n in `indices`, built lazily by calling `builder(index)`.
    `label` is a free-form description of the construction, kept for
    provenance when a sequence is logged/compared against others."""

    builder: callable
    indices: list
    label: str = ""
    _measures: dict = field(default_factory=dict, repr=False)
    _J_values: dict = field(default_factory=dict, repr=False)

    def measure_at(self, index):
        if index not in self._measures:
            self._measures[index] = self.builder(index)
        return self._measures[index]

    def J_at(self, index):
        if index not in self._J_values:
            self._J_values[index] = J(self.measure_at(index))
        return self._J_values[index]

    def J_values(self):
        """J(mu_index) for every index in self.indices, in order."""
        return [self.J_at(i) for i in self.indices]

    def trend(self, window=3):
        """Classify the tail of the sequence: 'rising' if each of the last
        `window` values strictly improves on the previous, 'plateaued' if
        the last `window` values are within 1e-6 of each other, else
        'mixed'. A cheap first pass for "is this construction still
        climbing or has it capped out" (KNOWLEDGE.md SS4's distinction
        between the still-rising d=3 growing-gadget construction and the
        provably-plateaued recursive blow-up)."""
        values = self.J_values()
        if len(values) < 2:
            return "insufficient-data"
        tail = values[-window:]
        if all(b > a for a, b in zip(tail, tail[1:])):
            return "rising"
        if max(tail) - min(tail) < 1e-6:
            return "plateaued"
        return "mixed"
