"""Parallel support-anchored cuts; update the actual remaining solid at each cut."""
from dataclasses import dataclass
import numpy as np

from .block import Frame, mesh_arrays
from .slab3d import SlabSpec
from .plane_clip import clip_z


def component_volumes(solid):
    return [float(p.volume()) for p in solid.decompose() if not p.is_empty()]


@dataclass
class ParallelPlan:
    frame: Frame
    lower: float
    upper: float
    thickness: float
    specs: list
    pieces: list
    steps: list
    remainder: object

    def to_dict(self):
        return {"kind": "parallel_support_start", "frame": self.frame.to_dict(),
                "normal": self.frame.basis[:, 2].tolist(), "support_lower_mm": self.lower,
                "support_upper_mm": self.upper, "thickness_mm": self.thickness,
                "initial_trim_allowed": False, "first_stage_kerf_mm": 0,
                "cuts": self.steps, "slab_count": len(self.specs),
                "cut_count": sum(s['cut_performed'] for s in self.steps),
                "remainder_volume_mm3": self.remainder.volume(),
                "remainder_component_volumes_mm3": component_volumes(self.remainder)}


def parallel_plan(block, normal, thickness):
    thickness = float(thickness)
    if not np.isfinite(thickness) or thickness <= 0:
        raise ValueError("Slab thickness must be positive")
    frame = Frame.from_dict({"normal": normal})
    remaining = frame.local_solid(block.solid)
    vertices, _ = mesh_arrays(remaining)
    lower, upper = float(vertices[:, 2].min()), float(vertices[:, 2].max())
    ratio = (upper - lower) / thickness
    count = int(np.floor(ratio + 1e-12 * max(1., ratio)))
    if count > 2000:
        raise ValueError("More than 2000 slabs per direction; increase thickness or split the job")
    specs, pieces, steps = [], [], []
    for i in range(count):
        spec = SlabSpec(frame, lower + i * thickness, lower + (i + 1) * thickness)
        before = remaining.volume()
        piece, piece_cap = clip_z(remaining, spec.upper, keep_above=False)
        after, after_cap = clip_z(remaining, spec.upper, keep_above=True)
        error = abs(before - piece.volume() - after.volume())
        if error > 1e-9 * max(1., before):
            raise ValueError("Sequential cut volume balance failed")
        cut = not piece.is_empty() and not after.is_empty()
        kind = "through_cut" if cut else "take_final_piece" if not piece.is_empty() else "empty_layer"
        steps.append({"slab_index": i, "kind": kind, "cut_performed": cut,
                      "plane_offset_mm": spec.upper,
                      "plane_normal_world": frame.basis[:, 2].tolist(),
                      "before_volume_mm3": before, "slab_volume_mm3": piece.volume(),
                      "after_volume_mm3": after.volume(), "balance_error_mm3": error,
                      "cap_roundoff_correction": {"slab": piece_cap, "remainder": after_cap},
                      "slab_component_volumes_mm3": component_volumes(piece),
                      "remaining_component_volumes_mm3": component_volumes(after)})
        specs.append(spec); pieces.append(piece); remaining = after
    return ParallelPlan(frame, lower, upper, thickness, specs, pieces, steps, remaining)
