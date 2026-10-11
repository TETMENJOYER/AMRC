"""CLI for stage 1: synthetic solid, one slab, and independent product trials."""
from __future__ import annotations

import argparse
import copy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np
from shapely.geometry import Polygon

from .block import Block
from .block_generator import demo_experiment
from .slab3d import SlabSpec, build_slab, footprint_geojson
from .verify3d import product_geometry, validate_product


def parse_inputs(data: dict):
    if data.get("schema_version") != 1 or data.get("kind") != "geometry_trials":
        raise ValueError("Expected schema_version = 1, kind = geometry_trials")
    if data.get("angle_unit") != "radians":
        raise ValueError("Expected angle_unit = radians")
    product = data["product"]
    if product.get("aspect_ratio", 3) != 3 or product.get("kerf", 0) != 0 or product.get("allowance", 0) != 0:
        raise ValueError("Stage 1 uses aspect ratio 3, zero kerf and zero allowance")
    short_side = float(product["short_side"])
    if not np.isfinite(short_side) or short_side <= 0:
        raise ValueError("short_side must be finite and positive")
    names = set()
    for trial in data["trials"]:
        if not isinstance(trial["name"], str) or not trial["name"] or trial["name"] in names:
            raise ValueError("Trial names must be nonempty and unique")
        if not isinstance(trial.get("expected_accepted"), bool):
            raise ValueError("Each independent trial needs a boolean expected_accepted")
        names.add(trial["name"])
    return Block.from_dict(data["block"]), SlabSpec.from_dict(data["slab"]), short_side


def run_experiment(inputs: dict):
    block, spec, short_side = parse_inputs(inputs)
    block_check = block.diagnostics()
    slab = build_slab(block, spec)
    trials = []
    for trial in inputs["trials"]:
        check = validate_product(block, spec, trial["state"], short_side, corner_diagnostics=True)
        rectangle = Polygon(product_geometry(spec, trial["state"], short_side)[2])
        in_footprint = slab.footprint.covers(rectangle)
        sampled = slab.sampled_intersection.covers(rectangle)
        trials.append({"name": trial["name"], "state": trial["state"],
                       "expected_accepted": trial["expected_accepted"],
                       "expectation_met": check["accepted"] == trial["expected_accepted"],
                       "fits_effective_region": in_footprint,
                       "fits_three_display_sections": sampled,
                       "footprint_agrees_with_volume": in_footprint == check["accepted"],
                       **check})
    canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    result = {"schema_version": 1, "kind": "stage1_geometry_result", "inputs": copy.deepcopy(inputs),
              "inputs_sha256": hashlib.sha256(canonical).hexdigest(), "block_check": block_check,
              "slab_geometry": slab.diagnostics, "effective_region": footprint_geojson(slab),
              "trial_results": trials,
              "all_expectations_met": all(t["expectation_met"] for t in trials),
              "all_footprint_checks_agree": all(t["footprint_agrees_with_volume"] for t in trials),
              "trials_are_independent": True, "packing_performed": False,
              "versions": {name: version(name) for name in ("numpy", "shapely", "manifold3d", "trimesh")}}
    return result, block, slab


def recheck_result(data: dict) -> dict:
    if data.get("schema_version") != 1 or data.get("kind") != "stage1_geometry_result":
        raise ValueError("Expected a stage1_geometry_result")
    inputs = data["inputs"]
    block, spec, short_side = parse_inputs(inputs)
    trials = []
    # Deliberately ignore saved effective_region and trial_results.
    for trial in inputs["trials"]:
        actual = validate_product(block, spec, trial["state"], short_side, corner_diagnostics=True)
        trials.append({"name": trial["name"], "expected_accepted": trial["expected_accepted"],
                       "expectation_met": actual["accepted"] == trial["expected_accepted"], **actual})
    return {"kind": "independent_geometry_trial_audit", "trials_are_independent": True,
            "all_expectations_met": all(t["expectation_met"] for t in trials), "trials": trials}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Generate and check the seeded nonconvex example")
    demo.add_argument("--seed", type=int, default=20261009)
    demo.add_argument("--output", type=Path, default=Path("outputs/geometry3d-demo"))
    run = commands.add_parser("run", help="Check a saved CSG experiment")
    run.add_argument("experiment", type=Path)
    run.add_argument("--output", type=Path, default=Path("outputs/geometry3d-custom"))
    verify = commands.add_parser("verify", help="Recompute trials from their original 3D inputs")
    verify.add_argument("result", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            audit = recheck_result(json.loads(args.result.read_text(encoding="utf-8-sig")))
            print(json.dumps(audit, ensure_ascii=True, indent=2, allow_nan=False))
            return 0 if audit["all_expectations_met"] else 2
        inputs = (demo_experiment(args.seed) if args.command == "demo" else
                  json.loads(args.experiment.read_text(encoding="utf-8-sig")))
        result, block, slab = run_experiment(inputs)
        from .report3d import write_geometry_report
        report = write_geometry_report(result, block, slab, args.output)
        print(json.dumps({"report": str(report.resolve()), "region": result["slab_geometry"],
                          "all_expectations_met": result["all_expectations_met"],
                          "all_footprint_checks_agree": result["all_footprint_checks_agree"]}, indent=2))
        return 0 if result["all_expectations_met"] and result["all_footprint_checks_agree"] else 2
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
