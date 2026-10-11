"""Finite recency memory invariant to rectangle labels and theta + pi."""
from __future__ import annotations

from collections import deque
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


def equivalent(a: np.ndarray, b: np.ndarray, position_tol: float, angle_tol: float) -> bool:
    if len(a) != len(b):
        return False
    if len(a) == 0:
        return True
    if np.linalg.norm(a[:, :2].mean(axis=0) - b[:, :2].mean(axis=0)) > position_tol:
        return False
    distance = np.linalg.norm(a[:, None, :2] - b[None, :, :2], axis=-1)
    delta = (a[:, None, 2] - b[None, :, 2] + np.pi / 2) % np.pi - np.pi / 2
    adjacency = (distance <= position_tol) & (np.abs(delta) <= angle_tol)
    if not adjacency.any(axis=0).all() or not adjacency.any(axis=1).all():
        return False
    # A full matching enforces the tolerance for EVERY rectangle. Averaged
    # assignment costs could hide a single large move in a large layout.
    matching = maximum_bipartite_matching(csr_matrix(adjacency), perm_type="column")
    return bool((matching >= 0).all())


class TabuMemory:
    def __init__(self, tenure: int = 32, position_tol: float = 0.01,
                 angle_tol: float = np.deg2rad(1.0)):
        if tenure < 1 or position_tol <= 0 or angle_tol <= 0:
            raise ValueError("Tabu parameters must be positive")
        self.tenure = tenure
        self.position_tol, self.angle_tol = position_tol, angle_tol
        self.entries: deque[tuple[int, np.ndarray]] = deque()

    def _expire(self, iteration: int):
        while self.entries and self.entries[0][0] <= iteration:
            self.entries.popleft()

    def contains(self, state: np.ndarray, iteration: int) -> bool:
        self._expire(iteration)
        return any(equivalent(state, old, self.position_tol, self.angle_tol)
                   for _, old in self.entries)

    def add(self, relaxed_state: np.ndarray, iteration: int):
        self._expire(iteration)
        self.entries.append((iteration + self.tenure, relaxed_state.copy()))

