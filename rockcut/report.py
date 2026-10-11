from __future__ import annotations

import csv
import html
import json
from pathlib import Path
import numpy as np
from .problem import Problem


def write_report(result: dict, output: str | Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as Patch

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with (output / "placements.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "center_x", "center_y", "angle_radians", "angle_degrees"])
        for i, (x, y, t) in enumerate(result["state"]):
            writer.writerow([i + 1, x, y, t, np.rad2deg(t)])
    problem = result["problem"]
    volume_note = ""
    alignment_note = ""
    if result.get('alignment'):
        a = result['alignment']
        alignment_note = (f'<p>Доводка после поиска: {a["elapsed_seconds"]:.2f} с; число изделий сохранено. '
                          f'Штраф аккуратности: {a["before"]["score"]:.5f} → {a["after"]["score"]:.5f}. '
                          'Исходные координаты и метрики до/после сохранены в JSON. '
                          'Метрика не является проверкой последовательности резов.</p>')
        if a.get('stop_reason'):
            reasons = {'plateau': 'улучшения затухли', 'regularity_target_reached': 'штраф практически равен нулю',
                       'round_limit': 'достигнут защитный предел раундов',
                       'time_safety_limit': 'достигнут защитный предел времени',
                       'single_pass_finished': 'завершён однократный проход'}
            alignment_note += (f'<p>Полных раундов: {a["completed_rounds"]}. Причина остановки: '
                               f'{html.escape(reasons.get(a["stop_reason"], a["stop_reason"]))}. '
                               'История изменения проверенной метрики сохранена в JSON.</p>')
        if a.get('phases'):
            alignment_note += ('<p>Сначала согласуются углы, затем кромки. Параллельные изделия '
                'используют общий параметр угла; на этапе кромок взаимные углы заморожены. '
                'Смешанный штраф выше приведён как диагностическая метрика; '
                'он не определяет приём кандидатов или остановку.</p>')
            for p in a['phases']:
                label = 'углы' if p['phase'] == 'angles' else 'кромки'
                alignment_note += (f'<p>Этап «{label}»: {p["completed_rounds"]} полных раундов, '
                    f'{html.escape(reasons.get(p["stop_reason"], p["stop_reason"]))}. '
                    f'Проверенная целевая метрика: {p["before"][p["metric"]]:.6g} → '
                    f'{p["after"][p["metric"]]:.6g}.</p>')
    if "validation3d" in result:
        audit = result["validation3d"]
        if not audit["accepted"]:
            raise ValueError("Cannot publish an unaccepted 3D packing")
        (output / "validation3d.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        volume_note = '<p>Полный объём каждого изделия проверен в исходном однородном 3D-блоке. ' \
                      '<a href="validation3d.json">Объёмная проверка</a>.</p>'
    fig, ax = plt.subplots(figsize=(11, 8), layout="constrained")
    fig.patch.set_facecolor("#f5f7fb")
    ax.set_facecolor("#f5f7fb")
    model = Problem.from_dict(problem)
    from .region_plot import paint_region
    paint_region(ax, model.physical_region, color="#e0e4eb", edge="#26334d", linewidth=2)
    for i, points in enumerate(result["validation"]["rectangle_vertices"]):
        ax.add_patch(Patch(points, facecolor="#78c6b0", edgecolor="#155c50", linewidth=0.9))
        x, y, theta = result["state"][i]
        ax.text(x, y, str(i + 1), ha="center", va="center", fontsize=9, color="#163c37")
    xmin, ymin, xmax, ymax = model.physical_region.bounds
    pad = max(xmax - xmin, ymax - ymin) * .055
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal")
    ax.set_xlabel(f"x, {problem['units']}")
    ax.set_ylabel(f"y, {problem['units']}")
    ax.set_title(f"{result['n']} изделий · использование {result['utilization']:.2%}\n"
                 f"Граница по площади: {result['area_upper_bound']} · геометрия проверена", loc="left", pad=16)
    ax.grid(alpha=0.15)
    fig.savefig(output / "layout.png", dpi=170)
    fig.savefig(output / "layout.svg")
    plt.close(fig)
    rows = "".join(f"<tr><td>{i+1}</td><td>{x:.5f}</td><td>{y:.5f}</td><td>{np.rad2deg(t):.5f}</td></tr>"
                   for i, (x, y, t) in enumerate(result["state"]))
    unit = html.escape(problem["units"])
    document = f"""<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Раскрой слэба — результат</title>
<style>body{{font:16px system-ui;background:#f5f7fb;color:#203047;max-width:1100px;margin:36px auto;padding:0 20px}}
h1{{font-size:30px}}.cards{{display:flex;gap:16px;flex-wrap:wrap}}.card{{background:white;border-radius:12px;padding:20px;min-width:160px}}
.card b{{display:block;font-size:30px;margin-top:6px}}img{{width:100%;margin:20px 0}}table{{border-collapse:collapse;background:white;width:100%}}
td,th{{padding:10px;text-align:right;border-bottom:1px solid #e0e4eb}}th{{text-align:right}}a{{color:#155c50}}</style>
<h1>Раскрой одного слэба</h1><p>{html.escape(problem['name'])} · изделие {problem['short_side']*3:g} × {problem['short_side']:g} {unit}</p>
<div class="cards"><div class="card">Найдено изделий<b>{result['n']}</b></div>
<div class="card">Использование площади<b>{result['utilization']:.2%}</b></div>
<div class="card">Верхняя граница по площади<b>{result['area_upper_bound']}</b></div></div>
<p>Геометрическая проверка пройдена. Остаток: {result['unused_area']:,.2f} {unit}².
Время поиска: {result.get('search_elapsed_seconds', result['elapsed_seconds']):.1f} с.</p>
{volume_note}
{alignment_note}
<p>Результат эвристического поиска. Доказанный оптимум не заявляется.
Пропил и припуски равны нулю; последовательность резов не проверяется.</p>
<img src="layout.svg" alt="Расположение изделий внутри слэба">
<p><a href="result.json">Полный результат JSON</a> · <a href="placements.csv">Координаты CSV</a> · <a href="layout.png">Изображение PNG</a></p>
<table><thead><tr><th>№</th><th>x, {unit}</th><th>y, {unit}</th><th>Угол, °</th></tr></thead><tbody>{rows}</tbody></table></html>"""
    (output / "report.html").write_text(document, encoding="utf-8")
    return output / "report.html"
