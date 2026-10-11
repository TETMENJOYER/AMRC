"""Render polygonal regions without importing optional 3D dependencies."""
from shapely.geometry.polygon import orient


def paint_region(ax, region, color="#cee5dc", edge="#245b4a", alpha=1., linewidth=1.5):
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    if region.is_empty:
        return
    polygons = region.geoms if region.geom_type == "MultiPolygon" else [region]
    for polygon in polygons:
        polygon = orient(polygon, sign=1)
        vertices, codes = [], []
        for ring in [polygon.exterior, *polygon.interiors]:
            coords = list(ring.coords)
            vertices.extend(coords)
            codes.extend([Path.MOVETO] + [Path.LINETO] * (len(coords) - 2) + [Path.CLOSEPOLY])
        ax.add_patch(PathPatch(Path(vertices, codes), facecolor=color, edgecolor=edge, alpha=alpha, lw=linewidth))
