"""Batched differentiable SAT/boundary penalties with independent Adam states."""
from __future__ import annotations

import math
import time
import numpy as np
import torch
from .problem import Problem


class BudgetExpired(Exception):
    pass


def resolve_device(requested: str) -> str:
    if requested == "auto":
        # Small default instances benefit more from avoiding launch overhead.
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested, but this PyTorch installation has no available CUDA device")
    if requested not in ("cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    return requested


class LocalOptimizer:
    def __init__(self, problem: Problem, device: str = "cpu", margin: float = 1e-5):
        self.device = resolve_device(device)
        self.margin = margin  # search clearance, not a physical kerf requirement
        self.normals = torch.tensor(problem.normals, dtype=torch.float64, device=self.device)
        self.offsets = torch.tensor(problem.offsets, dtype=torch.float64, device=self.device)
        self.obstacles = torch.tensor(problem.obstacle_triangles, dtype=torch.float64, device=self.device)
        if len(self.obstacles):
            edges = self.obstacles.roll(-1, dims=1) - self.obstacles
            axes = torch.stack((edges[..., 1], -edges[..., 0]), dim=-1)
            self.obstacle_axes = axes / axes.norm(dim=-1, keepdim=True)
        self._indices: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.pair_weights: torch.Tensor | None = None
        self.boundary_weights: torch.Tensor | None = None

    def violations(self, states: torch.Tensor, smooth: float = 0.0,
                   margin: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Unsigned pair depths and boundary violations, before squaring/weighting."""
        if states.ndim != 3 or states.shape[-1] != 3:
            raise ValueError("states must have shape (B, N, 3)")
        n = states.shape[1]
        if n == 0:
            return states[..., 0], states[..., :1].expand(-1, -1, len(self.offsets))
        margin = self.margin if margin is None else margin
        co, si = states[..., 2].cos(), states[..., 2].sin()
        u, v = torch.stack((co, si), -1), torch.stack((-si, co), -1)
        eps = smooth * 0.02

        def support_abs(x):
            return torch.sqrt(x.square() + eps**2) if eps else x.abs()

        ru = torch.einsum("bni,mi->bnm", u, self.normals)
        rv = torch.einsum("bni,mi->bnm", v, self.normals)
        support = 1.5 * support_abs(ru) + 0.5 * support_abs(rv)
        boundary = torch.einsum("bni,mi->bnm", states[..., :2], self.normals) + support - self.offsets
        boundary = torch.relu(boundary + margin)
        if len(self.obstacles):
            # Full rectangle versus each forbidden triangle. Corners alone can
            # miss an indentation or an excluded region wholly inside a product.
            axes = torch.cat((torch.stack((u, v), dim=2)[:, :, None].expand(-1, -1, len(self.obstacles), -1, -1),
                              self.obstacle_axes[None, None].expand(states.shape[0], n, -1, -1, -1)), dim=3)
            radius = (1.5 * support_abs((axes * u[:, :, None, None]).sum(-1))
                      + .5 * support_abs((axes * v[:, :, None, None]).sum(-1)))
            center = (axes * states[:, :, None, None, :2]).sum(-1)
            triangle = torch.einsum("bnkai,kvi->bnkav", axes, self.obstacles)
            gaps = torch.maximum(triangle.min(-1).values - center - radius,
                                 center - radius - triangle.max(-1).values)
            separation = (smooth * (torch.logsumexp(gaps / smooth, dim=-1) - math.log(5))
                          if smooth else gaps.max(-1).values)
            forbidden = torch.relu(margin - separation)
            boundary = torch.cat((boundary, forbidden), dim=-1)
        if n < 2:
            return states[:, :0, 0], boundary
        if n not in self._indices:
            ij = torch.triu_indices(n, n, 1, device=self.device)
            self._indices[n] = (ij[0], ij[1])
        i, j = self._indices[n]
        axes = torch.stack((u[:, i], v[:, i], u[:, j], v[:, j]), dim=2)
        ri = (1.5 * support_abs((axes * u[:, i, None]).sum(-1))
              + 0.5 * support_abs((axes * v[:, i, None]).sum(-1)))
        rj = (1.5 * support_abs((axes * u[:, j, None]).sum(-1))
              + 0.5 * support_abs((axes * v[:, j, None]).sum(-1)))
        difference = states[:, i, :2] - states[:, j, :2]
        projection = (axes * difference[:, :, None]).sum(-1)
        center_abs = torch.sqrt(projection.square() + eps**2) - eps if eps else projection.abs()
        gaps = center_abs - ri - rj
        # log-mean-exp is <= max: smoothing remains conservative with respect
        # to separation, and is removed completely during L-BFGS polishing.
        separation = (smooth * (torch.logsumexp(gaps / smooth, dim=-1) - math.log(4))
                      if smooth else gaps.max(dim=-1).values)
        return torch.relu(margin - separation), boundary

    def penalty(self, states: torch.Tensor, smooth: float = 0.0,
                margin: float | None = None) -> torch.Tensor:
        """One scalar per candidate; unweighted behavior is the original baseline."""
        pair, boundary = self.violations(states, smooth, margin)
        pair, boundary = pair.square(), boundary.square()
        if self.pair_weights is not None:
            pair = pair * self.pair_weights
            boundary = boundary * self.boundary_weights
        return boundary.sum(dim=(1, 2)) + pair.sum(dim=-1)

    def reset_weights(self):
        self.pair_weights = self.boundary_weights = None

    def update_weights(self, state: np.ndarray) -> float:
        """Boost recurring violated constraints, decay solved ones, cap at 8.

        Rectangle slots stay stable along a trajectory. Call reset_weights when
        starting a new N or replacing starts with independently labeled layouts.
        """
        with torch.no_grad():
            q = torch.tensor(state[None], dtype=torch.float64, device=self.device)
            pair, boundary = self.violations(q, margin=0)
            if self.pair_weights is None:
                self.pair_weights = torch.ones_like(pair[0])
                self.boundary_weights = torch.ones_like(boundary[0])
            self.pair_weights = (1 + 0.85 * (self.pair_weights - 1)
                                 + 1.5 * (pair[0] > 1e-3)).clamp_max(8)
            self.boundary_weights = (1 + 0.85 * (self.boundary_weights - 1)
                                     + 1.5 * (boundary[0] > 1e-3)).clamp_max(8)
            return max(float(self.pair_weights.max()) if self.pair_weights.numel() else 1,
                       float(self.boundary_weights.max()) if self.boundary_weights.numel() else 1)

    def relax(self, batch: np.ndarray, steps: int = 180, deadline: float = math.inf) -> np.ndarray:
        initial = torch.tensor(batch, dtype=torch.float64, device=self.device)
        if initial.shape[1] == 0 or steps == 0:
            return np.array(batch, copy=True)
        centers = initial[..., :2].clone().requires_grad_(True)
        angles = initial[..., 2:].clone().requires_grad_(True)
        optimizer = torch.optim.Adam([{"params": [centers], "lr": 0.045},
                                      {"params": [angles], "lr": 0.018}])
        best = initial.clone()
        with torch.no_grad():
            best_loss = self.penalty(best)
        for step in range(steps):
            if step % 10 == 0 and time.monotonic() >= deadline:
                break
            optimizer.zero_grad(set_to_none=True)
            q = torch.cat((centers, angles), dim=-1)
            temperature = 0.03 * (1 - step / max(1, steps - 1))**2
            loss = self.penalty(q, smooth=temperature)
            loss.sum().backward()
            # No clipping across the batch: each candidate is an independent run.
            with torch.no_grad():
                norm = torch.sqrt(centers.grad.square().sum((1, 2)) + angles.grad.square().sum((1, 2)))
                factor = (10 / norm.clamp_min(10)).reshape(-1, 1, 1)
                centers.grad.mul_(factor)
                angles.grad.mul_(factor)
            rate = 0.15 + 0.85 * (1 - step / max(1, steps - 1))
            optimizer.param_groups[0]["lr"] = 0.045 * rate
            optimizer.param_groups[1]["lr"] = 0.018 * rate
            optimizer.step()
            if step % 5 == 0 or step == steps - 1:
                with torch.no_grad():
                    now = torch.cat((centers, angles), -1)
                    value = self.penalty(now)
                    improve = torch.isfinite(value) & (value < best_loss)
                    best[improve] = now[improve]
                    best_loss[improve] = value[improve]
        result = best.detach().cpu().numpy()
        result[..., 2] %= np.pi
        return result

    def polish(self, state: np.ndarray, iterations: int = 100,
               deadline: float = math.inf) -> np.ndarray:
        if not len(state) or time.monotonic() >= deadline or iterations == 0:
            return state.copy()
        q = torch.tensor(state[None], dtype=torch.float64, device=self.device, requires_grad=True)
        optimizer = torch.optim.LBFGS([q], lr=1, max_iter=iterations, max_eval=iterations * 2,
                                      history_size=20, tolerance_grad=1e-10,
                                      tolerance_change=1e-14, line_search_fn="strong_wolfe")
        best, best_value = q.detach().clone(), float("inf")

        def closure():
            nonlocal best, best_value
            if time.monotonic() >= deadline:
                raise BudgetExpired
            optimizer.zero_grad(set_to_none=True)
            value = self.penalty(q).sum()
            if torch.isfinite(value) and value.item() < best_value:
                best, best_value = q.detach().clone(), value.item()
            value.backward()
            return value

        try:
            optimizer.step(closure)
        except BudgetExpired:
            pass
        result = best[0].cpu().numpy()
        result[:, 2] %= np.pi
        return result
