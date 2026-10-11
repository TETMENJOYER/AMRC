from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
import platform
from typing import Callable
import numpy as np
import torch
import scipy
import shapely
from .problem import Problem
from .geometry import area_score
from .local import LocalOptimizer, resolve_device
from .seeds import shelf_seeds, fill_gaps, insert_one, perturb, random_layout, destroy_repair
from .tabu import TabuMemory
from .verify import check_candidate, verified_subset


@dataclass
class SearchConfig:
    seed: int = 20261008
    seconds: float = 120
    seconds_per_n: float = 45
    batch_size: int = 12
    adam_steps: int = 180
    polish_iterations: int = 120
    polish_candidates: int = 3
    iterations_per_n: int = 100
    tabu_tenure: int = 32
    tabu_position: float = 0.01
    tabu_angle_degrees: float = 1.0
    search_margin: float = 1e-5
    device: str = "auto"
    threads: int = 1
    diverse_starts: bool = False
    group_repair: bool = False
    adaptive_penalties: bool = False
    polish_all: bool = False
    retry_failed_n: bool = False
    anneal_margin: bool = False
    alignment_seconds: float | None = 0.
    alignment_stop_mode: str = 'convergence'
    alignment_method_policy: str = 'adaptive'

    def __post_init__(self):
        if self.alignment_seconds is not None and (not math.isfinite(self.alignment_seconds) or self.alignment_seconds < 0):
            raise ValueError('alignment_seconds must be finite and nonnegative')
        if self.alignment_stop_mode not in ('convergence', 'time'):
            raise ValueError('alignment_stop_mode must be convergence or time')
        if self.alignment_method_policy not in ('always', 'adaptive'):
            raise ValueError('alignment_method_policy must be always or adaptive')
        for key in ("seconds", "seconds_per_n", "tabu_position", "tabu_angle_degrees", "search_margin"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ("batch_size", "adam_steps", "polish_iterations", "polish_candidates",
                    "iterations_per_n", "tabu_tenure", "threads"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for key in ("diverse_starts", "group_repair", "adaptive_penalties", "polish_all",
                    "retry_failed_n", "anneal_margin"):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f"{key} must be a boolean")


VARIANTS = {
    "baseline": {},
    "diverse": {"diverse_starts": True},
    "repair": {"group_repair": True},
    "adaptive": {"adaptive_penalties": True},
    "polish_all": {"polish_all": True},
    "combined": {"diverse_starts": True, "group_repair": True, "adaptive_penalties": True},
    "restarts": {"retry_failed_n": True},
    "anneal": {"anneal_margin": True},
}


def solve(problem: Problem, config: SearchConfig | None = None,
          progress: Callable[[dict], None] | None = None,
          target_n: int | None = None,
          acceptance_check: Callable[[np.ndarray], dict] | None = None,
          initial_state: np.ndarray | None = None) -> dict:
    """Maximize N heuristically, or attempt exactly target_n.

    Failure at N is recorded as unknown. It never tightens the area upper bound.
    """
    cfg = config or SearchConfig()
    if target_n is not None and (not isinstance(target_n, int) or target_n < 0):
        raise ValueError("target_n must be a nonnegative integer")
    if target_n is not None and target_n > problem.area_upper_bound:
        raise ValueError(f"target_n exceeds the area upper bound {problem.area_upper_bound}")
    started = time.monotonic()
    deadline = started + cfg.seconds
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads)
    device = resolve_device(cfg.device)
    local = LocalOptimizer(problem, device, cfg.search_margin)

    def check_layout(physical):
        return check_candidate(problem, physical, acceptance_check)
    history: list[dict] = []
    best = np.empty((0, 3))
    best_check = check_layout(problem.denormalize(best))
    best_partial = None
    attempts = []
    timings = {key: 0.0 for key in ("initialization", "proposals", "adam", "polish",
                                    "area_scoring", "selection_and_tabu", "adaptive_weights")}
    counters = {key: 0 for key in ("batches", "candidates", "polished", "polish_area_improved",
                                   "polish_area_worsened", "repair_moves", "restart_batches", "n_restarts")}

    def emit(event: str, **fields):
        entry = {"event": event, "elapsed": round(time.monotonic() - started, 3), **fields}
        history.append(entry)
        if progress:
            progress(entry)

    def retry_count(count: int) -> bool:
        if not cfg.retry_failed_n or time.monotonic() >= deadline:
            return False
        counters["n_restarts"] += 1
        emit("retry_count", n=count)
        return True

    emit("start", device=device, area_upper_bound=problem.area_upper_bound)
    if initial_state is not None:
        warm = np.asarray(initial_state, dtype=float)
        if warm.size == 0:
            warm = np.empty((0, 3))
        if warm.ndim != 2 or warm.shape[1] != 3 or not np.isfinite(warm).all():
            raise ValueError("initial_state must be a finite physical (N, 3) array")
        if target_n is not None:
            warm = warm[:target_n]
        original_count = len(warm)
        warm, check = verified_subset(warm, check_layout)
        if len(warm) != original_count:
            emit("seed_subset", source="warm_start", original_n=original_count, retained_n=len(warm))
        if check["accepted"] and len(warm) > len(best):
            best, best_check = problem.normalize(warm), check
            emit("feasible", n=len(best), source="verified_warm_start")
    seed_deadline = min(deadline, started + min(12, cfg.seconds * 0.15))
    seeds = (shelf_seeds(problem, limit=2, deadline=seed_deadline, margin=0)
             + shelf_seeds(problem, limit=2, deadline=seed_deadline, margin=cfg.search_margin))
    for seed in seeds[:4]:
        if time.monotonic() < seed_deadline:
            seed = fill_gaps(problem, seed, seed_deadline)
        if target_n is not None:
            seed = seed[:target_n]
        original_count = len(seed)
        physical, check = verified_subset(problem.denormalize(seed), check_layout)
        seed = problem.normalize(physical)
        if len(seed) != original_count:
            emit("seed_subset", source="shelf_initialization", original_n=original_count, retained_n=len(seed))
        if check["accepted"] and len(seed) > len(best):
            best, best_check = seed.copy(), check
            emit("feasible", n=len(best), source="shelf_initialization")
    catalog = []
    if cfg.diverse_starts:
        catalog = shelf_seeds(problem, limit=12, deadline=seed_deadline,
                              margin=cfg.search_margin, diverse=True)
    timings["initialization"] = time.monotonic() - started
    # A total budget includes initialization. Empty placement remains a valid fallback.
    n = target_n if target_n is not None else len(best) + 1
    active_n = None
    memory = None
    tabu_clock = 0
    count_record, count_record_score = None, math.inf
    n_attempts = 0
    while n <= problem.area_upper_bound and time.monotonic() < deadline:
        if target_n is not None and len(best) == target_n:
            break
        n_started = time.monotonic()
        n_deadline = min(deadline, n_started + cfg.seconds_per_n)
        local.reset_weights()
        local.margin = cfg.search_margin
        if n != active_n:
            active_n = n
            memory = TabuMemory(cfg.tabu_tenure, cfg.tabu_position, np.deg2rad(cfg.tabu_angle_degrees))
            tabu_clock = 0
            count_record, count_record_score = None, math.inf
            n_attempts = 0
        n_attempts += 1
        emit("attempt", n=n, attempt_for_n=n_attempts)

        def make_start(k):
            if cfg.diverse_starts and k % 4 == 2:
                return random_layout(problem, n, rng)
            if cfg.diverse_starts and k % 4 == 1 and catalog:
                eligible = [q for q in catalog if len(q) >= max(1, n - 5)]
                q = eligible[int(rng.integers(len(eligible)))].copy() if eligible else best.copy()
                if len(q) > n:
                    q = q[:n]
            else:
                q = best.copy()
            while len(q) < n and time.monotonic() < n_deadline:
                q = insert_one(problem, q, rng)
            if len(q) == n and k:
                q = perturb(problem, q, rng, strength=0.6)
            return q

        phase_started = time.monotonic()
        initial = []
        for k in range(cfg.batch_size):
            q = make_start(k)
            if len(q) != n:
                break
            initial.append(q)
        timings["proposals"] += time.monotonic() - phase_started
        if not initial:
            attempts.append({"n": n, "status": "not_found_within_budget", "iterations": 0,
                             "attempt_for_n": n_attempts, "seconds": time.monotonic() - n_started})
            if retry_count(n):
                continue
            break
        # Keep a fixed batch size even when insertion consumed the remaining budget.
        while len(initial) < cfg.batch_size:
            initial.append(initial[0].copy())
        current = initial[0].copy()
        record = current.copy()
        record_score = area_score(problem, record).total
        if count_record is not None and count_record_score < record_score:
            record, record_score = count_record.copy(), count_record_score
        stagnation = 0
        found = None
        total_tabu = 0
        completed_iterations = 0
        iteration = -1
        for iteration in range(cfg.iterations_per_n):
            if time.monotonic() >= n_deadline:
                break
            if cfg.anneal_margin:
                # Reach contact geometry during the last quarter of an attempt.
                # The exact validator still decides acceptance at every stage.
                fraction = max(iteration / max(1, cfg.iterations_per_n - 1),
                               (time.monotonic() - n_started) / max(1e-9, n_deadline - n_started))
                local.margin = cfg.search_margin * max(0.0, 1.0 - fraction / 0.75)**2
            phase_started = time.monotonic()
            if iteration == 0:
                batch = np.stack(initial)
            elif cfg.diverse_starts and iteration % 6 == 0:
                candidates = [current.copy()]
                for k in range(1, cfg.batch_size):
                    q = make_start(k)
                    candidates.append(q if len(q) == n else current.copy())
                batch = np.stack(candidates)
                # New starts have independent labels; old pair weights do not apply.
                local.reset_weights()
                counters["restart_batches"] += 1
            else:
                strength = min(2.5, 1 + stagnation / 12)
                candidates = [current.copy()]
                while len(candidates) < cfg.batch_size:
                    source = record if rng.random() < 0.3 else current
                    if cfg.group_repair and len(candidates) % 3 == 0:
                        candidates.append(destroy_repair(problem, source, rng, n_deadline))
                        counters["repair_moves"] += 1
                    else:
                        candidates.append(perturb(problem, source, rng, strength))
                batch = np.stack(candidates)
            timings["proposals"] += time.monotonic() - phase_started
            phase_started = time.monotonic()
            batch = local.relax(batch, cfg.adam_steps, n_deadline)
            timings["adam"] += time.monotonic() - phase_started
            completed_iterations += 1
            counters["batches"] += 1
            counters["candidates"] += len(batch)
            phase_started = time.monotonic()
            scores = [area_score(problem, q).total for q in batch]
            timings["area_scoring"] += time.monotonic() - phase_started
            phase_started = time.monotonic()
            polish_count = cfg.batch_size if cfg.polish_all else cfg.polish_candidates
            for idx in np.argsort(scores)[:polish_count]:
                if time.monotonic() >= n_deadline:
                    break
                polished = local.polish(batch[idx], cfg.polish_iterations, n_deadline)
                value = area_score(problem, polished).total
                counters["polished"] += 1
                counters["polish_area_improved"] += int(value < scores[idx] - 1e-10)
                counters["polish_area_worsened"] += int(value > scores[idx] + 1e-10)
                if value <= scores[idx]:
                    batch[idx], scores[idx] = polished, value
            timings["polish"] += time.monotonic() - phase_started
            phase_started = time.monotonic()
            old_record_score = record_score
            allowed = []
            tabu_hits = 0
            for idx in np.argsort(scores):
                q, value = batch[idx], scores[idx]
                if value < 1e-6:
                    check = check_layout(problem.denormalize(q))
                    if check["accepted"]:
                        found = (q.copy(), check)
                        break
                is_tabu = memory.contains(q, tabu_clock + iteration)
                # Aspiration uses the record from BEFORE examining this batch.
                aspiration = value < old_record_score - 1e-12
                if is_tabu and not aspiration:
                    tabu_hits += 1
                else:
                    allowed.append(idx)
                if value < record_score:
                    record, record_score = q.copy(), value
            total_tabu += tabu_hits
            timings["selection_and_tabu"] += time.monotonic() - phase_started
            if found:
                best, best_check = found
                emit("feasible", n=n, source="tabu_gradient", iteration=iteration)
                break
            improved = record_score < old_record_score - 1e-10
            stagnation = 0 if improved else stagnation + 1
            if allowed:
                idx = min(allowed, key=lambda k: scores[k])
                current = batch[idx].copy()
                memory.add(current, tabu_clock + iteration)
            else:
                # Do not clear recency memory or accept a forbidden state.
                # Larger moves are relaxed and checked on the next iteration.
                current = perturb(problem, record, rng, strength=3.0)
                emit("diversify", n=n, iteration=iteration)
            if tabu_hits > cfg.batch_size // 2:
                memory.tenure = min(3 * cfg.tabu_tenure, memory.tenure + 2)
            if cfg.adaptive_penalties:
                phase_started = time.monotonic()
                max_weight = local.update_weights(current)
                timings["adaptive_weights"] += time.monotonic() - phase_started
            else:
                max_weight = 1.0
            if iteration % 3 == 0 or improved:
                emit("iteration", n=n, iteration=iteration, best_area_penalty=record_score,
                     tabu_hits=tabu_hits, tabu_tenure=memory.tenure, max_constraint_weight=max_weight,
                     search_margin=local.margin, tabu_iteration=tabu_clock + iteration)
        tabu_clock += completed_iterations
        attempts.append({"n": n, "status": "feasible" if found else "not_found_within_budget",
                         "iterations": completed_iterations, "tabu_rejections": total_tabu,
                         "best_area_penalty": 0.0 if found else record_score,
                         "attempt_for_n": n_attempts, "final_search_margin": local.margin,
                         "seconds": time.monotonic() - n_started})
        if not found:
            count_record, count_record_score = record.copy(), record_score
            best_partial = {"n": n, "state": problem.denormalize(record).tolist(),
                            "area_penalty": record_score,
                            "validation": check_layout(problem.denormalize(record))}
            if retry_count(n):
                continue
            break
        best_partial = None
        if target_n is not None:
            break
        n += 1
    physical = problem.denormalize(best)
    # Rebuild and verify independently once more; only this check authorizes output.
    best_check = check_layout(physical)
    if not best_check["accepted"]:
        raise RuntimeError("Final independent verification rejected the saved layout")
    search_elapsed = time.monotonic()-started
    alignment = None
    if cfg.alignment_seconds is None or cfg.alignment_seconds:
        from .alignment import AlignmentConfig, align_layout
        aligned = align_layout(problem, physical,
            AlignmentConfig(seconds=cfg.alignment_seconds, device=device, stop_mode=cfg.alignment_stop_mode,
                            method_policy=cfg.alignment_method_policy), acceptance_check,
            progress=lambda row: emit('alignment_improved', **row))
        physical = np.asarray(aligned['state'], dtype=float).reshape(-1, 3)
        best_check, alignment = aligned['validation'], aligned['alignment']
        timings['alignment'] = alignment['elapsed_seconds']
    reaches_bound = len(best) == problem.area_upper_bound
    stop_reason = ("area_upper_bound_reached" if reaches_bound else
                   "target_reached" if target_n is not None and len(best) == target_n else
                   "search_budget_exhausted")
    emit("finished", n=len(best), stop_reason=stop_reason)
    return {
        "schema_version": 1, "problem": problem.to_dict(), "config": asdict(cfg),
        "device_used": device, "target_n": target_n, "stop_reason": stop_reason,
        "n": len(best), "state": physical.tolist(), "angle_unit": "radians",
        "area_upper_bound": problem.area_upper_bound,
        "count_gap_to_area_bound": problem.area_upper_bound - len(best),
        "area_bound_reached": reaches_bound,
        "optimality_proven": False,
        "optimality_note": ("Count meets the numerical area bound; geometry verified in float64, not formally certified."
                            if reaches_bound else "Best feasible layout found by a bounded heuristic; optimality is not proven."),
        "slab_area": problem.area, "product_area": problem.product_area,
        "used_area": len(best) * problem.product_area,
        "unused_area": problem.area - len(best) * problem.product_area,
        "utilization": len(best) * problem.product_area / problem.area,
        "validation": best_check, "attempts": attempts,
        "best_unaccepted_candidate": best_partial,
        "elapsed_seconds": time.monotonic() - started, "search_elapsed_seconds": search_elapsed, "history": history,
        "phase_seconds": timings, "search_counters": counters,
        **({'alignment': alignment} if alignment is not None else {}),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__,
                     "shapely": shapely.__version__, "geos": shapely.geos_version_string},
    }
