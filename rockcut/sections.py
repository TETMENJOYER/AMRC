"""Packing in the common footprint of parallel, registered convex sections.

No alignment or convex hull replacement is performed. Three sampled sections
alone do not describe the unsampled material between their heights.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import html
import json
import math
from pathlib import Path
import sys

import numpy as np
from shapely.geometry import Polygon

from .problem import Problem
from .verify import validate


@dataclass
class Section:
    z: float
    problem: Problem


class SlabSections:
    def __init__(self, sections: list[Section], name: str = "sampled_slab"):
        if len(sections) < 2:
            raise ValueError("At least two parallel sections are required")
        self.sections = list(sections)
        self.name = name
        heights = [section.z for section in sections]
        if not all(math.isfinite(z) for z in heights) or any(
                b <= a for a, b in zip(heights, heights[1:])):
            raise ValueError("Section heights must be finite and strictly increasing")
        first = sections[0].problem
        if any(not s.problem.is_convex for s in sections):
            raise ValueError("The sampled-section interface requires convex sections; use the full 3D slab pipeline")
        if any(s.problem.units != first.units or s.problem.short_side != first.short_side
               for s in sections):
            raise ValueError("Sections must use the same units and product dimensions")

    @classmethod
    def from_dict(cls, data: dict) -> "SlabSections":
        if data.get("coordinate_system") != "shared_xy":
            raise ValueError("coordinate_system must be shared_xy; sections are not aligned automatically")
        if data.get("between_sections") != "unknown":
            raise ValueError("This version supports between_sections = unknown only")
        shared = {key: data[key] for key in
                  ("short_side", "units", "aspect_ratio", "kerf", "allowance", "touching_allowed")
                  if key in data}
        sections = [Section(float(item["z"]), Problem.from_dict(
            {**shared, "name": item.get("name", f"section_{i}"), "vertices": item["vertices"]}))
            for i, item in enumerate(data["sections"])]
        return cls(sections, data.get("name", "sampled_slab"))

    @classmethod
    def load(cls, path: str | Path) -> "SlabSections":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8-sig")))

    def to_dict(self) -> dict:
        data = self.sections[0].problem.to_dict()
        data.pop("vertices")
        data.update(name=self.name, coordinate_system="shared_xy", between_sections="unknown",
                    sections=[{"z": s.z, "name": s.problem.name,
                               "vertices": s.problem.vertices.tolist()} for s in self.sections])
        return data

    def effective_problem(self) -> Problem:
        # Work near the origin, but preserve the relative offsets of all sections.
        reference = self.sections[0].problem.vertices[0]
        polygons = [Polygon(s.problem.vertices - reference) for s in self.sections]
        common = polygons[0]
        for polygon in polygons[1:]:
            common = common.intersection(polygon)
        if common.is_empty or common.geom_type != "Polygon" or common.area <= 0:
            raise ValueError("Sections have no common footprint with positive area")
        first = self.sections[0].problem
        return Problem(np.asarray(common.exterior.coords)[:-1] + reference,
                       first.short_side, self.name + "_effective", first.units)

    def summary(self) -> dict:
        effective = self.effective_problem()
        return {"method": "intersection_of_registered_sections",
                "effective_area": effective.area, "effective_vertices": len(effective.vertices),
                "section_areas": [{"name": s.problem.name, "z": s.z, "area": s.problem.area,
                                   "retained_fraction": effective.area / s.problem.area}
                                  for s in self.sections],
                "sampled_height_span": self.sections[-1].z - self.sections[0].z,
                "checked_heights": [s.z for s in self.sections],
                "continuous_thickness_verified": False,
                "between_sections": "unknown", "units": effective.units}


def validate_sections(model: SlabSections, physical_state: np.ndarray) -> dict:
    """Rebuild each original section and each rectangle in physical units."""
    checks = []
    for section in model.sections:
        check = validate(section.problem, physical_state)
        check.pop("rectangle_vertices")
        checks.append({"name": section.problem.name, "z": section.z, **check})
    return {"accepted": all(c["accepted"] for c in checks), "sections": checks,
            "scope": "supplied_sections_only", "continuous_thickness_verified": False}


def verify_result(data: dict) -> dict:
    if data.get("angle_unit") != "radians":
        raise ValueError("Expected angle_unit = radians")
    state = np.asarray(data["state"], dtype=float)
    if state.size == 0:
        state = np.empty((0, 3))
    if state.ndim != 2 or state.shape[1] != 3 or data["n"] != len(state):
        raise ValueError("Invalid placements or declared count")
    model = SlabSections.from_dict(data["section_model"])
    # Recompute the footprint: do not trust exported vertices or accepted flags.
    footprint = validate(model.effective_problem(), state)
    footprint.pop("rectangle_vertices")
    layers = validate_sections(model, state)
    return {"accepted": footprint["accepted"] and layers["accepted"],
            "effective_validation": footprint, "section_validation": layers}


def write_section_report(model: SlabSections, output: str | Path,
                         result: dict | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    effective = model.effective_problem()
    summary = model.summary()
    for name, data in (("sections.json", model.to_dict()), ("effective.json", effective.to_dict()),
                       ("geometry.json", summary)):
        (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    colors = ["#2166ac", "#d46a21", "#7545a0"]
    fig = plt.figure(figsize=(18, 6), layout="constrained")
    ax3 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132)
    axp = fig.add_subplot(133)
    for i, section in enumerate(model.sections):
        color = colors[i % len(colors)]
        vertices = section.problem.vertices
        xyz = np.column_stack((vertices, np.full(len(vertices), section.z)))
        ax3.add_collection3d(Poly3DCollection([xyz], facecolor=color, edgecolor=color, alpha=.24))
        closed = np.vstack((vertices, vertices[0]))
        ax2.plot(closed[:, 0], closed[:, 1], color=color, linewidth=2,
                 label=f"z = {section.z:g} ({len(vertices)} вершин)")
    points = np.vstack([s.problem.vertices for s in model.sections])
    ax3.set_xlim(points[:, 0].min(), points[:, 0].max())
    ax3.set_ylim(points[:, 1].min(), points[:, 1].max())
    ax3.set_zlim(model.sections[0].z, model.sections[-1].z)
    # Vertical exaggeration makes the layers readable; it is explicitly labeled.
    ax3.set_box_aspect((1.3, 1, .6))
    ax3.set_title("Сечения по высоте\nмасштаб z увеличен для наглядности")
    ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z")
    for axis in (ax2, axp):
        axis.add_patch(Patch(effective.vertices, facecolor="#d7e8df", edgecolor="#172f27", linewidth=2))
        axis.set_aspect("equal")
        axis.set_xlim(points[:, 0].min() - 50, points[:, 0].max() + 50)
        axis.set_ylim(points[:, 1].min() - 50, points[:, 1].max() + 50)
        axis.set_xlabel(f"x, {effective.units}"); axis.set_ylabel(f"y, {effective.units}")
        axis.grid(alpha=.15)
    ax2.set_title(f"Пересечение сечений\nплощадь {effective.area:,.0f} {effective.units}²")
    ax2.legend(loc="upper right", fontsize=8)
    if result is not None:
        checks = verify_result(result)
        if not checks["accepted"]:
            raise ValueError("Cannot report a layout that fails section validation")
        corners = validate(effective, np.asarray(result["state"]).reshape(-1, 3))["rectangle_vertices"]
        for i, vertices in enumerate(corners):
            axp.add_patch(Patch(vertices, facecolor="#78c6b0", edgecolor="#155c50", linewidth=.8))
            x, y, _ = result["state"][i]
            axp.text(x, y, str(i + 1), ha="center", va="center", fontsize=7)
        axp.set_title(f"{result['n']} изделий в общем контуре\nпроверены во всех сечениях")
    else:
        axp.set_title("Эффективный полигон для раскроя")
    fig.savefig(output / "sections.png", dpi=150)
    fig.savefig(output / "sections.svg")
    plt.close(fig)
    rows = "".join(f"<tr><td>{html.escape(s.problem.name)}</td><td>{s.z:g}</td>"
                   f"<td>{len(s.problem.vertices)}</td><td>{s.problem.area:,.2f}</td>"
                   f"<td>{effective.area / s.problem.area:.2%}</td></tr>" for s in model.sections)
    layout = (f"<p>Найдено <b>{result['n']} изделий</b>; использование площади общего контура "
              f"{result['utilization']:.2%}. Геометрическая проверка каждого сечения пройдена.</p>"
              '<p><a href="report.html">Координаты и отчёт раскладки</a> · '
              '<a href="result.json">Результат с исходными сечениями</a></p>') if result is not None else ""
    page = f'''<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Слэб переменной формы</title>
<style>body{{font:16px/1.5 system-ui;max-width:1500px;margin:32px auto;padding:0 20px;color:#203047;background:#f5f7fb}}
img{{width:100%}}table{{border-collapse:collapse;background:white}}td,th{{padding:10px 16px;border-bottom:1px solid #ddd}}
a{{color:#155c50}}.note{{padding:14px;background:#fff3da}}</style>
<h1>Общий контур слэба по нескольким сечениям</h1>
<p>{html.escape(model.name)}. Сечения заданы в общей системе координат x, y.
Эффективный полигон — их пересечение: {effective.area:,.2f} {html.escape(effective.units)}²,
{len(effective.vertices)} вершин.</p>{layout}
<img src="sections.svg" alt="Сечения по высоте, их пересечение и раскладка">
<table><tr><th>Сечение</th><th>Высота z</th><th>Вершин</th><th>Площадь</th><th>Общая часть / сечение</th></tr>{rows}</table>
<p>Доля общей части характеризует площадь, доступную по всем заданным высотам.
Она не является долей использованного объёма блока.</p>
<p class="note">Проверка распространяется на заданные сечения. Геометрия между ними неизвестна:
три измерения не исключают локального сужения между слоями. Оси изделий перпендикулярны плоскостям сечений;
наклон изделий по толщине не моделируется. Пропил и припуски пока равны нулю.</p>
<p><a href="effective.json">Эффективный полигон для обычного solve</a> ·
<a href="sections.json">Исходные сечения</a> · <a href="geometry.json">Геометрические показатели</a> ·
<a href="sections.png">Рисунок PNG</a></p></html>'''
    (output / "sections-report.html").write_text(page, encoding="utf-8")
    return output / "sections-report.html"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "solve"):
        command = commands.add_parser(name)
        command.add_argument("sections", type=Path)
        command.add_argument("--output", type=Path, default=Path("outputs/sections"))
        if name == "solve":
            command.add_argument("--seconds", type=float, default=30)
            command.add_argument("--seconds-per-n", type=float, default=30)
            command.add_argument("--seed", type=int, default=20261008)
    check = commands.add_parser("verify")
    check.add_argument("result", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            data = json.loads(args.result.read_text(encoding="utf-8-sig"))
            checks = verify_result(data)
            print(json.dumps(checks, ensure_ascii=True, indent=2, allow_nan=False))
            return 0 if checks["accepted"] else 2
        model = SlabSections.load(args.sections)
        result = None
        if args.command == "solve":
            from .solver import SearchConfig, solve
            from .report import write_report
            result = solve(model.effective_problem(), SearchConfig(
                seconds=args.seconds, seconds_per_n=args.seconds_per_n, seed=args.seed),
                progress=lambda event: print(json.dumps(event, ensure_ascii=True), flush=True))
            result["section_model"] = model.to_dict()
            result["section_geometry"] = model.summary()
            result["section_validation"] = validate_sections(model, np.asarray(result["state"]).reshape(-1, 3))
            if not verify_result(result)["accepted"]:
                raise ValueError("The layout failed independent section validation")
            write_report(result, args.output)
        report = write_section_report(model, args.output, result)
        print(f"Report: {report.resolve()}")
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
