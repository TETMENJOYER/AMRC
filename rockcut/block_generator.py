"""Seeded homogeneous blocks with a nonconvex exterior and no cavities."""
import numpy as np


def demo_experiment(seed: int = 20261009) -> dict:
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    top_shift = [float(rng.uniform(35, 55)), float(rng.uniform(-15, -5))]
    block = {
        "schema_version": 1, "units": "mm", "name": "homogeneous_nonconvex_block", "seed": seed,
        "base": {"type": "prism", "height": 450, "top_shift": top_shift,
                 "vertices": [[0,100],[200,0],[1000,0],[850,400],[1100,700],[550,850],[0,650]]}}
    return {
        "schema_version": 1, "kind": "geometry_trials", "block": block,
        "slab": {"frame": {"origin": [0, 0, 0], "normal": [0, 0, 1]}, "lower": 200, "upper": 230},
        "product": {"short_side": 100, "aspect_ratio": 3, "kerf": 0, "allowance": 0},
        "angle_unit": "radians",
        "trials": [
            {"name": "valid_lower", "state": [300, 250, 0], "expected_accepted": True},
            {"name": "valid_rotated", "state": [700, 300, float(np.deg2rad(20))], "expected_accepted": True},
            {"name": "valid_left", "state": [400, 450, 0], "expected_accepted": True},
            {"name": "crosses_concave_boundary", "state": [975, 450, 0], "expected_accepted": False},
            {"name": "valid_upper", "state": [650, 650, 0], "expected_accepted": True},
            {"name": "outside_outer_surface", "state": [80, 350, 0], "expected_accepted": False}
        ]}
