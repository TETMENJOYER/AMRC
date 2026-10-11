"""Reports for independent geometric trials; no optimized packing is implied."""
from __future__ import annotations

import html
import json
from pathlib import Path
import numpy as np
from shapely.geometry import mapping

from .block import mesh_arrays
from .verify3d import product_geometry
from .region_plot import paint_region


LABELS = {"valid_lower": "Допустимое изделие", "valid_rotated": "Допустимое с поворотом",
          "valid_left": "Допустимое слева", "crosses_concave_boundary": "Пересекает вогнутую границу",
          "valid_upper": "Допустимое сверху", "outside_outer_surface": "Выходит за внешнюю границу"}


def dump(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_obj(path: Path, solid):
    vertices, faces = mesh_arrays(solid)
    lines = ["# rockcut stage1, world coordinates in mm, float64"]
    lines.extend("v " + " ".join(format(float(x), ".17g") for x in v) for v in vertices)
    lines.extend("f " + " ".join(str(int(i) + 1) for i in f) for f in faces)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def write_geometry_report(result, block, slab, output) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dump(output / "result.json", result)
    dump(output / "experiment.json", result["inputs"])
    dump(output / "block.json", result["inputs"]["block"])
    dump(output / "effective.geojson", result["effective_region"])
    dump(output / "sampled-intersection.geojson", {"type": "Feature", "geometry": mapping(slab.sampled_intersection),
         "properties": {"diagnostic_only": True, "sample_heights": slab.sample_heights}})
    save_obj(output / "block.obj", block.solid)
    save_obj(output / "slab.obj", slab.spec.frame.world_solid(slab.solid))
    local_vertices, _ = mesh_arrays(slab.local_block)
    lo, hi = local_vertices.min(0), local_vertices.max(0)
    span = hi - lo

    def configure(ax, extra=None):
        bounds_lo, bounds_hi = lo[:2].copy(), hi[:2].copy()
        if extra is not None:
            bounds_lo = np.minimum(bounds_lo, np.min(extra, axis=0))
            bounds_hi = np.maximum(bounds_hi, np.max(extra, axis=0))
        padding = (bounds_hi - bounds_lo) * .05
        ax.set_aspect("equal")
        ax.set_xlim(bounds_lo[0] - padding[0], bounds_hi[0] + padding[0])
        ax.set_ylim(bounds_lo[1] - padding[1], bounds_hi[1] + padding[1])
        ax.set_xlabel("x слэба, мм"); ax.set_ylabel("y слэба, мм")
        ax.grid(alpha=.12)

    fig = plt.figure(figsize=(14, 6.5), layout="constrained")
    fig.get_layout_engine().set(wspace=.15)
    ax = fig.add_subplot(121, projection="3d")
    for solid, color, alpha in ((slab.local_block, "#587d91", .10), (slab.solid, "#cb713c", .58)):
        v, f = mesh_arrays(solid)
        ax.add_collection3d(Poly3DCollection(v[f], facecolor=color, edgecolor=color, linewidth=.35, alpha=alpha))
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(span)
    ax.set_xlabel("x слэба, мм"); ax.set_ylabel("y слэба, мм"); ax.set_zlabel("z, мм")
    ax.set_title(f"Невыпуклый блок и выбранный слэб\nz = {slab.spec.lower:g}…{slab.spec.upper:g} мм", pad=18)
    ax.view_init(elev=26, azim=-58)
    ax2 = fig.add_subplot(122)
    paint_region(ax2, slab.sampled_intersection, color="#f5c8b6", edge="#b44f2b")
    paint_region(ax2, slab.footprint)
    configure(ax2)
    ax2.set_title("Область для изделия полной толщины\nзелёная — допустима; оранжевая — ошибка трёх сечений", pad=14)
    fig.savefig(output / "overview.png", dpi=170)
    fig.savefig(output / "overview.svg")
    plt.close(fig)

    trials = result["trial_results"]
    if trials:
        fig, axes = plt.subplots((len(trials) + 2) // 3, 3, figsize=(15, 5 * ((len(trials) + 2) // 3)),
                                 squeeze=False, layout="constrained")
        for ax, trial in zip(axes.flat, trials):
            paint_region(ax, slab.footprint)
            _, _, rectangle = product_geometry(slab.spec, trial["state"], result["inputs"]["product"]["short_side"])
            ax.add_patch(Patch(rectangle, facecolor="#247d60" if trial["accepted"] else "#cf523d", alpha=.7,
                               edgecolor="#243d35", linewidth=1.5))
            configure(ax, rectangle)
            label = LABELS.get(trial["name"], trial["name"])
            ax.set_title(label + f"\nВне камня: {trial['missing_volume_mm3']:,.0f} мм³")
        for ax in list(axes.flat)[len(trials):]:
            ax.set_visible(False)
        fig.savefig(output / "trials.png", dpi=150)
        fig.savefig(output / "trials.svg")
        plt.close(fig)
    rows = []
    for trial in trials:
        cells = [LABELS.get(trial["name"], trial["name"]),
                 "допустимо" if trial["accepted"] else "недопустимо",
                 f"{trial['missing_volume_mm3']:,.2f}",
                 "да" if trial["fits_three_display_sections"] else "нет",
                 "да" if trial["all_corners_strictly_inside_diagnostic"] else "нет",
                 "совпало" if trial["expectation_met"] else "ОШИБКА"]
        rows.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>")
    info = slab.diagnostics
    all_ok = result["all_expectations_met"] and result["all_footprint_checks_agree"]
    doc = f'''<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Невыпуклый слэб — геометрия</title>
<style>body{{font:16px/1.55 system-ui;max-width:1280px;margin:32px auto;padding:0 22px;color:#203047;background:#f5f7fb}}
h1{{font-size:30px}}img{{width:100%}}table{{border-collapse:collapse;background:white;width:100%}}
th,td{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}.note{{background:#fff1d4;padding:15px}}
a{{color:#155c50}}code{{background:#e6eaf0;padding:2px 5px}}</style>
<h1>Первый этап: геометрия невыпуклого слэба</h1>
<p>Блок состоит из сплошного однородного камня, без внутренних полостей и сквозных отверстий.
Невыпуклость относится к внешней форме. В демонстрационном блоке нет боковых карманов.</p>
<p><b>{'Все контрольные случаи пройдены' if all_ok else 'Обнаружено расхождение'}.</b>
Проверено {len(trials)} отдельных изделий. Доступная область построена по всей толщине,
затем каждое изделие независимо проверено вычитанием исходного блока из его полного объёма.</p>
<p class="note">Это отдельные геометрические испытания, не совместная раскладка и не результат оптимизации количества изделий.</p>
<img src="overview.svg" alt="Невыпуклый блок, слэб и безопасная область">
<p>Толщина: {slab.spec.thickness:g} мм. Общая область: <b>{info['area_mm2']:,.2f} мм²</b>;
компонент: {info['components']}, отверстий: {info['holes']}.
Пересечение трёх сечений добавило бы <b>{info['unsafe_sampled_area_mm2']:,.2f} мм²</b> недопустимой площади.</p>
<table><tr><th>Испытание</th><th>Полный объём</th><th>Вне камня, мм³</th><th>Проходит три сечения</th>
<th>Все 8 углов внутри</th><th>Ожидание</th></tr>{''.join(rows)}</table>
{'<img src="trials.svg" alt="Шесть независимых пробных изделий">' if trials else ''}
<p>Проверка углов приведена как вспомогательная диагностика и не определяет допустимость.
Любой непустой остаток объёмного вычитания отклоняется; малый объём не считается допустимым по допуску.</p>
<p>Объём слэба: {info['slab_volume_mm3']:,.2f} мм³. Ошибка баланса «низ + слэб + верх»:
{info['partition_error_mm3']:.3g} мм³. Точность вычислений — float64 и численные геометрические библиотеки;
общий CSG-движок проверяется на аналитических тестах.</p>
<p><a href="experiment.json">Параметры опыта</a> · <a href="result.json">Результат и мировые координаты</a> ·
<a href="effective.geojson">Допустимая область</a> · <a href="block.obj">3D-блок OBJ</a> ·
<a href="slab.obj">3D-слэб OBJ</a> · <a href="overview.png">Рисунок PNG</a></p>
<p>Повторная проверка: <code>python -m rockcut.geometry3d verify result.json</code>.
Команда заново строит исходное тело по CSG-рецепту и игнорирует сохранённую 2D-область и флаги допустимости.</p>
</html>'''
    (output / "report.html").write_text(doc, encoding="utf-8")
    return output / "report.html"
