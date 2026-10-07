#!/usr/bin/env python3
"""(Re)compute the ``box_gripper_graspable`` label for the object datasets.

An object is labelled graspable when the smallest dimension of its visual
mesh's axis-aligned bounding box fits within the gripper width — i.e. a
parallel-jaw ("box") gripper can close its jaws across the object's thinnest
axis.  This is the deterministic reduction of
:meth:`ManipulatableObject.is_box_gripper_graspable`; the mesh-based
antipodal sampler in that method is intentionally *not* used here because it
samples the surface randomly (non-reproducible) and spurious near-antipodal
pairs on noisy scanned meshes make nearly everything pass.

The label is stored as checked-in data that the prepare scripts bake into
each dataset's ``catalog.json``:

* YCB — written as a ``box_gripper_graspable`` tag directly into the curated
  ``scripts/data/ycb_catalog.json``.
* GSO — written as a list of graspable object ids to
  ``scripts/data/gso_graspable.json`` (consumed by ``prepare_gso_assets.py``).
* ABO — written as a list of graspable object ids to
  ``scripts/data/abo_graspable.json`` (consumed by ``prepare_abo_assets.py``,
  which also extends it inline on every packaging run).

Usage::

    # YCB meshes live in the local assets folder once prepared/downloaded
    python scripts/compute_graspable.py ycb \
        --assets-dir unienv_genesis_collection/assets/objects/ycb

    # GSO meshes from a prepared tree (see prepare_gso_assets.py)
    python scripts/compute_graspable.py gso --assets-dir /tmp/gso_assets/objects/gso

    # ABO meshes from a prepared tree (see prepare_abo_assets.py)
    python scripts/compute_graspable.py abo --assets-dir /tmp/abo_assets/objects/abo
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Robotiq 2F-85 maximum opening; matches the cluttered-table example gripper.
GRIPPER_WIDTH = 0.085
TAG = "box_gripper_graspable"

_DATA_DIR = Path(__file__).parent / "data"
_YCB_CATALOG = _DATA_DIR / "ycb_catalog.json"
_GSO_GRASPABLE = _DATA_DIR / "gso_graspable.json"
_ABO_GRASPABLE = _DATA_DIR / "abo_graspable.json"

# Visual-mesh filename per dataset.
_MESH_NAME = {"ycb": "textured.obj", "gso": "model.obj", "abo": "model.obj"}


def mesh_min_extent(mesh_file: Path) -> float:
    """Smallest axis-aligned bounding-box dimension of *mesh_file* (metres)."""
    import trimesh

    mesh = trimesh.load(str(mesh_file), force="mesh")
    return float(min(mesh.extents))


def _is_graspable(args: tuple[str, Path, float]) -> tuple[str, bool | None]:
    object_id, mesh_file, gripper_width = args
    try:
        return object_id, mesh_min_extent(mesh_file) < gripper_width
    except Exception:  # unreadable / degenerate mesh
        return object_id, None


def compute(
    assets_dir: Path, mesh_name: str, gripper_width: float, workers: int,
) -> dict[str, bool | None]:
    object_ids = sorted(p.name for p in assets_dir.iterdir() if p.is_dir())
    if not object_ids:
        raise FileNotFoundError(f"No object directories found in {assets_dir}")
    tasks = [(oid, assets_dir / oid / mesh_name, gripper_width) for oid in object_ids]

    results: dict[str, bool | None] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for i, (oid, g) in enumerate(executor.map(_is_graspable, tasks)):
            results[oid] = g
            if (i + 1) % 200 == 0:
                print(f"  ... {i + 1}/{len(tasks)}")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", choices=("ycb", "gso", "abo"))
    parser.add_argument(
        "--assets-dir", type=Path, required=True,
        help="Directory of per-object subdirectories holding the visual meshes.",
    )
    parser.add_argument("--gripper-width", type=float, default=GRIPPER_WIDTH)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args(argv)

    results = compute(
        args.assets_dir.expanduser().resolve(),
        _MESH_NAME[args.dataset],
        args.gripper_width,
        args.workers,
    )
    graspable = sorted(k for k, v in results.items() if v is True)
    errors = sorted(k for k, v in results.items() if v is None)
    print(f"{args.dataset}: {len(graspable)}/{len(results)} graspable, "
          f"{len(errors)} mesh errors")
    if errors:
        print(f"  mesh errors (treated as not graspable): {errors[:10]}")

    if args.dataset == "gso":
        _GSO_GRASPABLE.write_text(json.dumps(graspable, indent=1) + "\n")
        print(f"Wrote {_GSO_GRASPABLE} ({len(graspable)} ids).")
    elif args.dataset == "abo":
        _ABO_GRASPABLE.write_text(json.dumps(graspable, indent=1) + "\n")
        print(f"Wrote {_ABO_GRASPABLE} ({len(graspable)} ids).")
    else:
        catalog = json.loads(_YCB_CATALOG.read_text())
        graspable_set = set(graspable)
        for object_id, entry in catalog.items():
            tags = [t for t in entry["tags"] if t != TAG]
            if object_id in graspable_set:
                tags.append(TAG)
            entry["tags"] = sorted(tags)
        _YCB_CATALOG.write_text(
            json.dumps(catalog, indent=1, ensure_ascii=False) + "\n"
        )
        print(f"Updated {_YCB_CATALOG} ({len(graspable)} tagged).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
