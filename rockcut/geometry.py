"""Fast scoring in normalized coordinates; no feasibility decisions here."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import shapely
from .problem import Problem


def corners(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=float).reshape(-1, 3)
    c, s = np.cos(state[:, 2]), np.sin(state[:, 2])
    # Canonical right angles have exact axis directions in floating-point geometry.
    c[np.abs(c) < 4 * np.finfo(float).eps] = 0
    s[np.abs(s) < 4 * np.finfo(float).eps] = 0
    u = np.column_stack((c, s)) * 1.5
    v = np.column_stack((-s, c)) * 0.5
    return np.stack([state[:, :2] - u - v, state[:, :2] + u - v,
                     state[:, :2] + u + v, state[:, :2] - u + v], axis=1)


@dataclass
class Score:
    overlap: float
    outside: float
    per_rectangle: np.ndarray

    @property
    def total(self) -> float:
        return (self.overlap + self.outside) / 3.0


def area_score(problem: Problem, state: np.ndarray) -> Score:
    n = len(state)
    if n == 0:
        return Score(0.0, 0.0, np.empty(0))
    polygons = shapely.polygons(corners(state))
    outside = np.asarray(shapely.area(shapely.difference(polygons, problem.polygon)))
    i, j = np.triu_indices(n, 1)
    intersections = shapely.area(shapely.intersection(polygons[i], polygons[j]))
    per = outside.copy()
    np.add.at(per, i, intersections)
    np.add.at(per, j, intersections)
    return Score(float(np.sum(intersections)), float(np.sum(outside)), per)

