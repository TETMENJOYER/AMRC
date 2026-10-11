"""Paired, time-budgeted comparisons with raw results and independent validation."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import html
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import random
import re
import statistics
import time
import numpy as np
import torch

from .problem import Problem
from .local import LocalOptimizer
from .seeds import shelf_seeds, fill_gaps
from .solver import SearchConfig, VARIANTS, solve
from .verify import validate


LABELS = {"baseline": "Исходный", "diverse": "Разные старты", "repair": "Перестройка группы",
          "adaptive": "Адаптивные штрафы", "polish_all": "L-BFGS для всех",
          "combined": "Объединённый", "restarts": "Повторные попытки", "anneal": "Уменьшение зазора"}


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def warm_worker():
    # One compute thread per worker; cold optimizer imports are outside all timers.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    p = Problem([[0, 0], [10, 0], [10, 10], [0, 10]], 1)
    local = LocalOptimizer(p)
    q = np.array([[[0, 0, 0.2], [1.8, 0.4, 0.6]]])
    local.polish(local.relax(q, 5)[0], 3)


def reference_layout(problem: Problem) -> np.ndarray:
    """Original deterministic initializer, with no heuristic search or time race."""
    candidates = shelf_seeds(problem, limit=2, margin=0) + shelf_seeds(problem, limit=2)
    best = np.empty((0, 3))
    for q in candidates:
        q = fill_gaps(problem, q)
        if len(q) > len(best) and validate(problem, problem.denormalize(q))["accepted"]:
            best = q
    return best


def run_job(job: dict) -> dict:
    cfg = SearchConfig(seed=job["seed"], seconds=job["seconds"],
                       seconds_per_n=job.get("seconds_per_n", job["seconds"]),
                       iterations_per_n=10000, device="cpu", threads=1, **VARIANTS[job["variant"]])
    result = solve(Problem.from_dict(job["case"]), cfg)
    # Reparse the same serialization used by downstream consumers.
    result = json.loads(json.dumps(result, allow_nan=False))
    check = validate(Problem.from_dict(result["problem"]), result["state"])
    if not check["accepted"] or check["n"] != result["n"]:
        raise RuntimeError("Benchmark rejected an exported placement")
    result["benchmark"] = {key: job[key] for key in ("case_id", "variant", "seed", "seconds", "reference_n", "target_n")}
    result["benchmark"]["independently_rechecked"] = True
    return result


def result_row(result: dict) -> dict:
    info = result["benchmark"]
    events = [e for e in result["history"] if e["event"] == "feasible"]
    target_times = [e["elapsed"] for e in events if e["n"] >= info["target_n"]]
    best_times = [e["elapsed"] for e in events if e["n"] >= result["n"]]
    success = result["n"] >= info["target_n"]
    partial = result["best_unaccepted_candidate"]
    penalty = (0.0 if success else partial["area_penalty"]
               if partial is not None and partial["n"] == info["target_n"] else None)
    return {
        **info, "n": result["n"], "success": success,
        "improvement_over_reference": result["n"] - info["reference_n"],
        "area_upper_bound": result["area_upper_bound"], "utilization": result["utilization"],
        "elapsed_seconds": result["elapsed_seconds"],
        "time_to_target": min(target_times) if target_times else None,
        "time_to_best": min(best_times) if best_times else 0.0,
        "best_penalty_at_target": penalty,
        "tabu_rejections": sum(a.get("tabu_rejections", 0) for a in result["attempts"]),
        "batches": result["search_counters"]["batches"],
        "polish_area_worsened": result["search_counters"]["polish_area_worsened"],
        "polished": result["search_counters"]["polished"],
        "phase_seconds": result["phase_seconds"],
    }


def summarize(rows: list[dict], variants: list[str], case_ids: list[str]) -> dict:
    by_case, overall = [], []
    baseline = {(r["case_id"], r["seed"]): r for r in rows if r["variant"] == "baseline"}
    for variant in variants:
        selected = [r for r in rows if r["variant"] == variant]
        if not selected:
            continue
        differences = [r["n"] - baseline[(r["case_id"], r["seed"])]["n"] for r in selected
                       if (r["case_id"], r["seed"]) in baseline]
        overall.append({"variant": variant, "runs": len(selected), "total_n": sum(r["n"] for r in selected),
                        "successes": sum(r["success"] for r in selected),
                        "mean_n": statistics.mean(r["n"] for r in selected),
                        "paired_wins": sum(d > 0 for d in differences),
                        "paired_ties": sum(d == 0 for d in differences),
                        "paired_losses": sum(d < 0 for d in differences),
                        "paired_mean_gain": statistics.mean(differences) if differences else None})
        for case_id in case_ids:
            group = [r for r in selected if r["case_id"] == case_id]
            if not group:
                continue
            times = [r["time_to_target"] for r in group if r["success"]]
            by_case.append({"case_id": case_id, "variant": variant, "runs": len(group),
                            "min_n": min(r["n"] for r in group), "max_n": max(r["n"] for r in group),
                            "mean_n": statistics.mean(r["n"] for r in group),
                            "successes": len(times), "target_n": group[0]["target_n"],
                            "median_success_time": statistics.median(times) if times else None,
                            "mean_elapsed": statistics.mean(r["elapsed_seconds"] for r in group),
                            "ert_seconds": (sum(r["time_to_target"] if r["success"] else r["elapsed_seconds"]
                                                for r in group) / len(times)) if times else None})
    return {"overall": overall, "by_case": by_case}


def write_summary(output: Path, manifest: dict, results: list[dict]) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .report import write_report

    rows = [result_row(r) for r in results]
    variants = manifest["variants"]
    cases = manifest["cases"]
    summary = summarize(rows, variants, [c["id"] for c in cases])
    save_json(output / "summary.json", summary)
    save_json(output / "metrics.json", rows)
    scalar_rows = [{k: v for k, v in r.items() if k != "phase_seconds"} for r in rows]
    with (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scalar_rows[0]))
        writer.writeheader()
        writer.writerows(scalar_rows)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for ax, case in zip(axes.flat, cases):
        stats = [next(s for s in summary["by_case"] if s["case_id"] == case["id"] and s["variant"] == v)
                 for v in variants]
        means = np.array([s["mean_n"] for s in stats])
        low, high = np.array([s["min_n"] for s in stats]), np.array([s["max_n"] for s in stats])
        colors = ["#64748b" if v == "baseline" else "#198675" for v in variants]
        ax.bar(np.arange(len(variants)), means, color=colors, yerr=[means - low, high - means], capsize=3)
        for i, s in enumerate(stats):
            ax.text(i, s["max_n"] + 0.1, f"{s['successes']}/{s['runs']}", ha="center", fontsize=10)
        ref = manifest["references"][case["id"]]
        ax.axhline(ref["target_n"], color="#c55332", linestyle="--", linewidth=1)
        ax.set_ylim(max(0, min(low) - 2), max(max(high), ref["target_n"]) + 1)
        ax.set_xticks(np.arange(len(variants)), [LABELS[v] for v in variants], rotation=28, ha="right", fontsize=8)
        ax.set_title(case["name"])
        ax.set_ylabel("Допустимых изделий")
        ax.grid(axis="y", alpha=0.15)
    for ax in list(axes.flat)[len(cases):]:
        ax.set_visible(False)
    fig.suptitle("Среднее число изделий; отрезки — минимум и максимум\nПодписи — число достижений общей цели; пунктир — цель", fontsize=13)
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)

    best_links = []
    for case in cases:
        eligible = [r for r in results if r["benchmark"]["case_id"] == case["id"]]
        winner = min(eligible, key=lambda r: (-r["n"], result_row(r)["time_to_best"]))
        write_report(winner, output / "best" / case["id"])
        best_links.append(f"<a href='best/{case['id']}/report.html'>{html.escape(case['name'])}: {winner['n']} изделий</a>")

    lines = ["# Сравнение вариантов плоского раскроя", "",
             f"Бюджет: {manifest['seconds']} с на запуск, случайные старты {manifest['seeds']}; "
             f"CPU, {manifest['workers']} одновременных процессов по одному вычислительному потоку.", "",
             f"Бюджет одной попытки для фиксированного N: {manifest['common_config']['seconds_per_n']:g} с. "
             "Число повторных попыток зависит от выбранной стратегии.", "",
             "Цель для каждого полигона зафиксирована до эксперимента: на одно изделие больше исходной "
             "инициализации, с ограничением сверху площадью. Поиск продолжает увеличивать N до исчерпания общего бюджета.", "",
             "Все экспортированные раскладки повторно проверены независимым валидатором. "
             "Оптимальность не заявляется. Серия исследовательская: три случайных старта не дают надёжной оценки "
             "вероятности успеха для произвольного слэба.", "",
             "| Вариант | Сумма изделий | Цель достигнута | Выигрыш / ничья / проигрыш к исходному |",
             "|---|---:|---:|---:|"]
    for s in summary["overall"]:
        lines.append(f"| {LABELS[s['variant']]} | {s['total_n']} | {s['successes']}/{s['runs']} | "
                     f"{s['paired_wins']} / {s['paired_ties']} / {s['paired_losses']} |")
    lines += ["", "Сравнения попарные: одинаковый полигон и случайный старт. Сумма изделий — по всем запускам, "
              "она не является результатом раскроя одного блока.", "",
              "| Полигон | Вариант | Изделий, среднее [мин; макс] | Цель N | Успех | Медиана времени успеха, с |",
              "|---|---|---:|---:|---:|---:|"]
    names = {c["id"]: c["name"] for c in cases}
    for s in summary["by_case"]:
        t = "—" if s["median_success_time"] is None else f"{s['median_success_time']:.2f}"
        lines.append(f"| {names[s['case_id']]} | {LABELS[s['variant']]} | {s['mean_n']:.2f} [{s['min_n']}; {s['max_n']}] | "
                     f"{s['target_n']} | {s['successes']}/{s['runs']} | {t} |")
    lines += ["", "Время успеха приведено только для успешных запусков; неудачи не превращаются в успешные "
              "наблюдения длительностью бюджета. В summary.json также сохранён ERT: суммарное время попыток "
              "до цели, включая неудачи, делённое на число успехов; при отсутствии успехов значение null.", "",
              "Профили времени — в metrics.json. Время загрузки библиотек и создания отчётов исключено; "
              "инициализация раскладки включена. Все процессы одинаково прогревают PyTorch перед измерением. "
              "Одновременные процессы делят ресурсы машины; порядок вариантов перемешан. Фактические длительности сохранены.", "",
              "Исходные настройки алгоритма сохранены под именем baseline. Все экспериментальные изменения отключены по умолчанию."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    overall_rows = "".join(f"<tr><td>{LABELS[s['variant']]}</td><td>{s['total_n']}</td>"
                           f"<td>{s['successes']}/{s['runs']}</td><td>{s['paired_wins']} / {s['paired_ties']} / {s['paired_losses']}</td></tr>"
                           for s in summary["overall"])
    doc = f"""<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Сравнение алгоритмов раскроя</title><style>body{{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#203047;background:#f5f7fb}}
table{{width:100%;border-collapse:collapse;background:white}}td,th{{padding:12px;border-bottom:1px solid #ddd;text-align:left}}img{{width:100%}}a{{color:#155c50}}li{{margin:10px 0}}</style>
<h1>Сравнение алгоритмов плоского раскроя</h1><p>{len(results)} запусков · {manifest['seconds']:g} с на запуск · {manifest['common_config']['seconds_per_n']:g} с на попытку N · {len(cases)} полигона · {len(manifest['seeds'])} случайных старта</p>
<p>Принимаются только независимо проверенные допустимые раскладки. Общая цель — на одно изделие больше исходной инициализации.
Серия исследовательская; оптимальность не заявляется.</p><table><tr><th>Вариант</th><th>Сумма изделий</th><th>Цель достигнута</th><th>Выигрыш / ничья / проигрыш</th></tr>{overall_rows}</table>
<p>Сумма — по всем запускам. Сравнения выполнены попарно на одинаковых полигонах и случайных стартах.</p>
<img src="comparison.png" alt="Сравнение допустимых раскладок по полигонам">
<h2>Лучшие найденные раскладки</h2><ul>{''.join('<li>'+s+'</li>' for s in best_links)}</ul>
<p><a href="report.md">Подробный отчёт и методика</a> · <a href="metrics.csv">Все измерения CSV</a> · <a href="summary.json">Сводка JSON</a> · <a href="manifest.json">Условия эксперимента</a></p>
<p>Медианы времени успеха в подробном отчёте считаются только по успешным попыткам. Неудачные попытки сохранены отдельно.
В каждой попытке время ограничено, поэтому неудача не доказывает невозможность размещения.</p></html>"""
    (output / "report.html").write_text(doc, encoding="utf-8")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("benchmarks/suite.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark"))
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--seconds-per-n", type=float, help="Budget per count attempt; defaults to --seconds")
    parser.add_argument("--seeds", default="101,202,303")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cases", help="Comma-separated case IDs; default: all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]
    variants = args.variants.split(",")
    if len(set(seeds)) != len(seeds) or len(set(variants)) != len(variants):
        parser.error("seeds and variants must be unique")
    if args.workers < 1 or args.workers > 8 or not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("positive seconds and 1..8 workers required")
    if any(v not in VARIANTS for v in variants) or any(s < 0 for s in seeds):
        parser.error("invalid variant or seed")
    seconds_per_n = args.seconds if args.seconds_per_n is None else args.seconds_per_n
    if not np.isfinite(seconds_per_n) or seconds_per_n <= 0:
        parser.error("seconds-per-n must be finite and positive")
    cases = json.loads(args.suite.read_text(encoding="utf-8-sig"))["cases"]
    if args.cases:
        selected = set(args.cases.split(","))
        cases = [c for c in cases if c["id"] in selected]
        if len(cases) != len(selected):
            parser.error("unknown case ID")
    if not cases or len(cases) > 4:
        parser.error("suite must contain between one and four cases")
    if len({c["id"] for c in cases}) != len(cases) or any(not re.fullmatch(r"[a-z0-9_]+", c["id"]) for c in cases):
        parser.error("case IDs must be unique lowercase names")
    args.output.mkdir(parents=True, exist_ok=True)
    references = {}
    for case in cases:
        p = Problem.from_dict(case)
        q = reference_layout(p)
        references[case["id"]] = {"reference_n": len(q), "target_n": min(len(q) + 1, p.area_upper_bound),
                                   "area_upper_bound": p.area_upper_bound,
                                   "state": p.denormalize(q).tolist()}
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))}
    manifest = {"schema_version": 1, "seconds": args.seconds, "seeds": seeds, "variants": variants,
                "variant_settings": {v: VARIANTS[v] for v in variants}, "workers": args.workers,
                "cases": cases, "references": references, "source_hashes": source_hashes,
                "machine": {"cpu": platform.processor(), "logical_cpus": os.cpu_count(), "platform": platform.platform()},
                "versions": {"python": platform.python_version(),
                             **{name: version(name) for name in ("torch", "numpy", "scipy", "shapely", "matplotlib")}},
                "common_config": SearchConfig(seconds=args.seconds, seconds_per_n=seconds_per_n, iterations_per_n=10000).__dict__}
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            parser.error("output already contains an experiment; choose another directory or --resume")
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            parser.error("Cannot resume: source, suite, machine, or configuration differs")
    else:
        save_json(manifest_path, manifest)
    jobs = [{"case": case, "case_id": case["id"], "variant": variant, "seed": seed,
             "seconds": args.seconds, "seconds_per_n": seconds_per_n,
             **{k: references[case["id"]][k] for k in ("reference_n", "target_n")}}
            for case in cases for seed in seeds for variant in variants]
    random.Random(20261008).shuffle(jobs)
    results, pending = [], []
    for job in jobs:
        path = args.output / "runs" / f"{job['case_id']}_{job['seed']}_{job['variant']}.json"
        if path.exists() and args.resume:
            result = json.loads(path.read_text(encoding="utf-8"))
            if not validate(Problem.from_dict(result["problem"]), result["state"])["accepted"]:
                raise RuntimeError(f"Invalid cached layout: {path}")
            results.append(result)
        else:
            pending.append((job, path))
    started = time.monotonic()
    print(json.dumps({"event": "benchmark_start", "runs": len(jobs), "pending": len(pending),
                      "seconds_per_run": args.seconds, "workers": args.workers,
                      "targets": {k: v["target_n"] for k, v in references.items()}}), flush=True)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    with ProcessPoolExecutor(max_workers=args.workers, initializer=warm_worker) as pool:
        futures = {pool.submit(run_job, job): path for job, path in pending}
        for future in as_completed(futures):
            result = future.result()
            save_json(futures[future], result)
            results.append(result)
            row = result_row(result)
            print(json.dumps({"event": "run_done", "done": len(results), "total": len(jobs),
                              **{k: row[k] for k in ("case_id", "variant", "seed", "n", "success", "time_to_target")},
                              "wall_seconds": round(time.monotonic() - started, 1)}), flush=True)
    results.sort(key=lambda r: (r["benchmark"]["case_id"], r["benchmark"]["seed"], r["benchmark"]["variant"]))
    summary = write_summary(args.output, manifest, results)
    print(json.dumps({"event": "benchmark_done", "wall_seconds": round(time.monotonic() - started, 1),
                      "overall": summary["overall"], "report": str((args.output / "report.html").resolve())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
