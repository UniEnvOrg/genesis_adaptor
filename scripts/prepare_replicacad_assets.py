#!/usr/bin/env python3
"""Prepare ReplicaCAD assets for the collection's Hugging Face assets repo.

Downloads (or reuses) the `ai-habitat/ReplicaCAD_dataset
<https://huggingface.co/datasets/ai-habitat/ReplicaCAD_dataset>`_ dataset and
reorganizes its articulated furniture and stage shells into the layout
expected by
``unienv_genesis_collection.objects.hub_objects.articulated.replicacad_furniture``::

    <output>/objects/replicacad/LICENSE                  # upstream license file, verbatim
    <output>/objects/replicacad/README.md                # provenance + attribution
    <output>/objects/replicacad/catalog.json             # curated annotations + tags
    <output>/objects/replicacad/fridge/fridge.urdf       # authored URDF (convex box collisions)
    <output>/objects/replicacad/fridge/*.glb             # GLB visual meshes
    <output>/scenes/rooms/replicacad/<stage_id>/<stage_id>.glb
    <output>/scenes/rooms/replicacad/arrangements/<scene_id>.scene_instance.json

Upstream URDF mesh references are directory-local already (``filename="body.glb"``);
a rewrite pass keeps them directory-local while normalizing the upstream
**comma-separated numeric vector attributes** (e.g. ``scale="1, 1, 1"``) to
whitespace-separated ones — ``genesis.ext.urdfpy`` (which both the Genesis URDF
loader and the offline FK helper use) parses vector attributes with
``np.fromstring(value, sep=" ")`` and fails on the commas.

Every repackaged item is sanity-loaded in Genesis on GPU — loader acceptance,
a non-empty joint set, and the offline-FK union bbox cross-checked against
``entity.get_vAABB()`` at q=0 — plus one render per item for a texture
eyeball.  Failing items are dropped with a warning instead of failing the
whole run.

Usage::

    python scripts/prepare_replicacad_assets.py --output-dir /tmp/replicacad_assets
    # or with an already-downloaded dataset snapshot:
    python scripts/prepare_replicacad_assets.py --source <snapshot-dir> --output-dir /tmp/replicacad_assets

Then upload to the Hub (needs write access to the org)::

    hf upload UniEnvOrg/unienv_genesis_collection_assets \\
        /tmp/replicacad_assets/objects/replicacad objects/replicacad --repo-type dataset
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HF_DATASET_REPO = "ai-habitat/ReplicaCAD_dataset"

# The dataset's URDF directory: one sub-directory per articulated furniture
# item (fridge, cabinets, drawers, doors, …).
URDF_DIR = "urdf"
STAGE_DIR = "stages"
ARRANGEMENT_DIR = "configs/scenes"
UPSTREAM_LICENSE_FILE = "LICENSE.txt"

# ---------------------------------------------------------------------------
# Item selection: the dataset's articulated furniture subset.  Keys are the
# **upstream articulated-object template names** (what the
# ``configs/scenes/*.scene_instance.json`` arrangement files reference as
# ``template_name``), so the arrangement converter can resolve catalog ids
# without an alias map.
# ---------------------------------------------------------------------------
ARTICULATED_ITEMS: Dict[str, Tuple[str, str]] = {
    # item id            urdf dir              urdf file name
    "cabinet": ("cabinet", "cabinet.urdf"),
    "chestOfDrawers_01": ("chest_of_drawers", "chestOfDrawers_01.urdf"),
    "door1": ("doors", "door1.urdf"),
    "door2": ("doors", "door2.urdf"),
    "door3": ("doors", "door3.urdf"),
    "doubledoor": ("doors", "doubledoor.urdf"),
    "fridge": ("fridge", "fridge.urdf"),
    "kitchen_counter": ("kitchen_counter", "kitchen_counter.urdf"),
    "kitchenCupboard_01": ("kitchen_cupboards", "kitchenCupboard_01.urdf"),
}

# Upstream also ships ``<name>_dynamic.urdf`` twins; they differ from the
# plain URDFs *only* in habitat-specific ``<collision group="2">`` markup
# (verified by diff), which Genesis ignores.  The plain (static) URDF is
# shipped under the template name.

# Curated per-item metadata (annotations + tags), shipped with the assets as
# ``objects/replicacad/catalog.json``.  ``standing_quaternion`` is the upright
# placement orientation: the URDF base-link frames follow the per-item
# ReplicaCAD authoring convention (see README), so each entry carries the
# minimal rotation mapping the item's up axis (measured on the offline FK
# union bbox: ``+y`` for the fridge/cabinet/counter/doors, ``+z`` for
# ``chestOfDrawers_01`` / ``kitchenCupboard_01``) onto Genesis's ``+z``.
_STANDING_QUAT_Y_UP = (0.7071067811865476, 0.7071067811865476, 0.0, 0.0)  # +y -> +z
_STANDING_QUAT_Z_UP = (1.0, 0.0, 0.0, 0.0)  # already +z

CATALOG: Dict[str, dict] = {
    "cabinet": {
        "language_annotations": ["furniture", "a cabinet", "a sliding-door cabinet"],
        "tags": ["furniture", "articulated", "cabinet"],
        "standing_quaternion": _STANDING_QUAT_Y_UP,
    },
    "chestOfDrawers_01": {
        "language_annotations": ["furniture", "a chest of drawers", "a tall chest of drawers"],
        "tags": ["furniture", "articulated", "drawer"],
        "standing_quaternion": _STANDING_QUAT_Z_UP,
    },
    "door1": {
        "language_annotations": ["furniture", "a door", "a paneled interior door"],
        "tags": ["furniture", "articulated", "door"],
        "standing_quaternion": _STANDING_QUAT_Y_UP,
    },
    "door2": {
        "language_annotations": ["furniture", "a door", "a glazed interior door"],
        "tags": ["furniture", "articulated", "door"],
        "standing_quaternion": _STANDING_QUAT_Y_UP,
    },
    "door3": {
        "language_annotations": ["furniture", "a door", "a glazed interior door"],
        "tags": ["furniture", "articulated", "door"],
        "standing_quaternion": _STANDING_QUAT_Y_UP,
    },
    "doubledoor": {
        "language_annotations": ["furniture", "a double door", "a glazed double door"],
        "tags": ["furniture", "articulated", "door"],
        "standing_quaternion": _STANDING_QUAT_Y_UP,
    },
    "fridge": {
        "language_annotations": ["furniture", "a fridge", "a two-door refrigerator"],
        "tags": ["furniture", "articulated", "kitchen", "appliance"],
        "standing_quaternion": _STANDING_QUAT_Y_UP,
    },
    "kitchen_counter": {
        "language_annotations": ["furniture", "a kitchen counter", "a kitchen counter with drawers"],
        "tags": ["furniture", "articulated", "kitchen", "drawer"],
        "standing_quaternion": _STANDING_QUAT_Y_UP,
    },
    "kitchenCupboard_01": {
        "language_annotations": ["furniture", "a kitchen cupboard", "a kitchen cupboard wall unit"],
        "tags": ["furniture", "articulated", "kitchen", "cabinet"],
        "standing_quaternion": _STANDING_QUAT_Z_UP,
    },
}

PROVENANCE_README = """\
# ReplicaCAD articulated furniture + stage shells

Repackaged from the [ReplicaCAD dataset](https://huggingface.co/datasets/ai-habitat/ReplicaCAD_dataset)
(© Meta/Facebook Reality Labs; artist recreation of the scanned "FRL apartment"
variations from the Replica dataset): the **articulated furniture** URDFs
(fridge, cabinets, drawers, doors) with their GLB visual meshes, plus the
**stage shells** (apartment / staging-room GLBs) and the `scene_instance.json`
arrangement files, ingested by
`genesis_adaptor/scripts/prepare_replicacad_assets.py`.

> **License note.** The upstream dataset is inconsistent about its license:
> its Hugging Face dataset card (front matter and prose) declares **CC BY 4.0**,
> while the `LICENSE.txt` file shipped in the snapshot states **CC BY-NC 4.0**
> with a non-commercial clause. The upstream `LICENSE.txt` is shipped verbatim
> as `LICENSE` — consumers should treat the more restrictive of the two
> (CC BY-NC 4.0, no commercial use) as governing and attribute
> "ReplicaCAD dataset, © Meta/Facebook Reality Labs (CC BY-NC 4.0)" when using
> these models.

* One self-contained directory per item under `objects/replicacad/<item_id>/`;
  the URDFs keep their **authored convex box collision geometry** as-is.
* Item ids are the upstream articulated-object template names, so the
  `configs/scenes/*.scene_instance.json` arrangements (shipped verbatim under
  `scenes/rooms/replicacad/arrangements/`) reference them directly.
* The upstream `<name>_dynamic.urdf` twins were not shipped: they differ from
  the plain URDFs only in habitat-specific `<collision group="2">` markup,
  which Genesis ignores.
* The upstream comma-separated numeric vector attributes
  (`scale="1, 1, 1"` — unparseable by `genesis.ext.urdfpy`) were normalized to
  whitespace-separated ones; habitat-specific `*.ao_config.json` ambient
  occlusion sidecars were dropped.
* The URDF base-link frames follow the per-item ReplicaCAD authoring
  conventions (height along `+y` for most items, along `+z` for
  `chestOfDrawers_01` / `kitchenCupboard_01`); the catalog carries the matching
  `standing_quaternion` for upright placement in a Z-up Genesis world.

When using these models, please attribute the ReplicaCAD dataset
— see `LICENSE` and the license note above.
"""

# Fallback for a hypothetical catalog entry without ``standing_quaternion``
# (all curated entries carry one): the common (+y-up) ReplicaCAD convention.
_PROBE_STANDING_QUAT = _STANDING_QUAT_Y_UP


# ---------------------------------------------------------------------------
# URDF rewriting
# ---------------------------------------------------------------------------

def normalize_vector_attribs(text: str) -> str:
    """Normalize comma-separated numeric vector attributes to space-separated.

    Upstream URDFs contain values like ``scale="1, 1, 1"`` and
    ``size="0.617 0.145, 0.894"``.  ``genesis.ext.urdfpy`` parses every
    numeric attribute with ``np.fromstring(value, sep=" ")`` — the commas make
    the URDF load (in Genesis and in the offline FK helper) fail with
    ``ValueError: string or file could not be read to its end due to
    unmatched data``.  Quoted values that consist only of digits, signs,
    dots, exponents and whitespace, with at least one comma, get their commas
    replaced by single spaces.
    """
    def _fix(match: "re.Match[str]") -> str:
        name, value = match.group(1), match.group(2)
        if "," in value and re.fullmatch(r"[\s\d.,eE+-]+", value):
            value = re.sub(r"\s*,\s*", " ", value).strip()
        return f'{name}="{value}"'

    return re.sub(r'([A-Za-z_:][\w:.-]*)="([^"]*)"', _fix, text)


def rewrite_mesh_paths(text: str, urdf_dir: str) -> str:
    """Rewrite ``filename="…"`` mesh references to be directory-local.

    Upstream ReplicaCAD URDFs already reference their meshes
    directory-locally (``filename="body.glb"``); the rewrite is defensive and
    strips a leading ``urdf/<urdf_dir>/`` or ``<urdf_dir>/`` prefix, mirroring
    the YCB pipeline's rewrite step, should an upstream variant ship
    repo-relative references.
    """
    prefix = re.escape(urdf_dir)
    return re.sub(rf'(filename=")(?:urdf/)?(?:{prefix}/)?', r"\1", text)


def referenced_mesh_files(urdf_file) -> List[str]:
    """Distinct ``filename=`` references of a URDF (mirrors ``xml_file_references``)."""
    root = ET.parse(str(urdf_file)).getroot()
    return list(dict.fromkeys(
        value
        for element in root.iter()
        if (value := element.get("filename")) is not None
    ))


def _verify_item_dir(obj_out_dir: Path, item_id: str) -> None:
    """Check that every mesh referenced by the rewritten URDF exists."""
    urdf = obj_out_dir / f"{item_id}.urdf"
    for filename in referenced_mesh_files(urdf):
        if not (obj_out_dir / filename).is_file():
            raise FileNotFoundError(f"{urdf.name} references missing mesh: {filename}")


def repackage_item(source: Path, objects_root: Path, item_id: str) -> Path:
    """Repackage one articulated furniture item into ``objects_root/<item_id>/``."""
    urdf_dir, urdf_name = ARTICULATED_ITEMS[item_id]
    src_dir = source / URDF_DIR / urdf_dir
    if not (src_dir / urdf_name).is_file():
        raise FileNotFoundError(f"Missing upstream URDF: {src_dir / urdf_name}")

    obj_out_dir = objects_root / item_id
    obj_out_dir.mkdir(parents=True, exist_ok=True)

    # URDF with normalized vector attributes and directory-local mesh paths,
    # renamed to the descriptor's `<item_id>.urdf` convention.
    text = rewrite_mesh_paths(
        normalize_vector_attribs((src_dir / urdf_name).read_text()), urdf_dir
    )
    (obj_out_dir / f"{item_id}.urdf").write_text(text)

    # Every referenced mesh/texture file, copied verbatim (the authored convex
    # box collisions are primitive <box> geoms in the URDF itself — no mesh
    # files for them).
    for filename in referenced_mesh_files(obj_out_dir / f"{item_id}.urdf"):
        src_file = src_dir / filename
        if not src_file.is_file():
            raise FileNotFoundError(f"{urdf_name} references missing mesh: {filename}")
        shutil.copy2(src_file, obj_out_dir / filename)

    _verify_item_dir(obj_out_dir, item_id)
    return obj_out_dir


def prepare(source: Path, output_dir: Path, item_ids: Sequence[str]) -> Tuple[Path, Path]:
    """Repackage the dataset into ``output_dir``; return (objects, scenes) roots."""
    for required in (URDF_DIR, STAGE_DIR, ARRANGEMENT_DIR, UPSTREAM_LICENSE_FILE):
        if not (source / required).exists():
            raise FileNotFoundError(f"No '{required}' in source: {source}")

    objects_root = output_dir / "objects" / "replicacad"
    scenes_root = output_dir / "scenes" / "rooms" / "replicacad"
    objects_root.mkdir(parents=True, exist_ok=True)
    scenes_root.mkdir(parents=True, exist_ok=True)

    for item_id in item_ids:
        repackage_item(source, objects_root, item_id)
    print(f"Prepared {len(item_ids)} items in {objects_root}")

    # Stage shells: scenes/rooms/replicacad/<stage_id>/<stage_id>.glb
    n_stages = 0
    for glb in sorted((source / STAGE_DIR).glob("*.glb")):
        stage_dir = scenes_root / glb.stem
        stage_dir.mkdir(exist_ok=True)
        shutil.copy2(glb, stage_dir / glb.name)
        n_stages += 1
    print(f"Prepared {n_stages} stage shells in {scenes_root}")

    # Arrangements: the upstream scene_instance.json files, shipped verbatim.
    arrangements_dir = scenes_root / "arrangements"
    arrangements_dir.mkdir(exist_ok=True)
    n_arrangements = 0
    for instance_file in sorted((source / ARRANGEMENT_DIR).glob("*.scene_instance.json")):
        shutil.copy2(instance_file, arrangements_dir / instance_file.name)
        n_arrangements += 1
    print(f"Prepared {n_arrangements} arrangement files in {arrangements_dir}")

    shutil.copy2(source / UPSTREAM_LICENSE_FILE, objects_root / "LICENSE")
    shutil.copy2(source / UPSTREAM_LICENSE_FILE, scenes_root / "LICENSE")
    (objects_root / "README.md").write_text(PROVENANCE_README)
    (scenes_root / "README.md").write_text(PROVENANCE_README)
    _write_catalog(objects_root, item_ids)

    return objects_root, scenes_root


def _write_catalog(objects_root: Path, item_ids: Sequence[str]) -> None:
    """Write the curated catalog, cross-checked against the prepared items.

    Every shipped item **must** have a curated entry — the descriptors raise
    ``KeyError`` on ids missing from the catalog, so a mismatch is a packaging
    bug and fails the pipeline instead of silently shipping unusable ids.

    Incremental runs (``--items`` subset into an already-populated output
    tree) are supported: entries of items that are *not* part of this run are
    kept from the existing ``catalog.json`` when present.
    """
    prepared = list(item_ids)
    missing_entries = [i for i in prepared if i not in CATALOG]
    if missing_entries:
        raise KeyError(
            f"No curated catalog entry for item(s): {missing_entries}. "
            "Add them to CATALOG in scripts/prepare_replicacad_assets.py."
        )
    catalog_file = objects_root / "catalog.json"
    catalog: Dict[str, dict] = {}
    if catalog_file.is_file():
        existing = json.loads(catalog_file.read_text())
        catalog = {k: v for k, v in existing.items() if k not in set(prepared)}
    catalog.update({item_id: CATALOG[item_id] for item_id in prepared})
    for entry in catalog.values():
        assert entry["language_annotations"] and entry["tags"] is not None
    catalog_file.write_text(
        json.dumps(catalog, indent=1, ensure_ascii=False) + "\n"
    )
    print(f"Catalog cross-check OK ({len(catalog)} items).")


# ---------------------------------------------------------------------------
# Genesis sanity-load gate (GPU)
# ---------------------------------------------------------------------------

def _quat_rotate(quat: Sequence[float], vector: Sequence[float]) -> List[float]:
    """Rotate *vector* by the (w, x, y, z) quaternion *quat*."""
    w, x, y, z = quat
    vx, vy, vz = vector
    # t = 2 * q_xyz × v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def _world_aabb_at_standing(item, standing_quat: Sequence[float]) -> Tuple[List[float], List[float]]:
    """Offline union AABB (8 corners) rotated by *standing_quaternion* — world frame."""
    local = item.get_local_bounding_box()
    low = local.low.detach().cpu().numpy()
    high = local.high.detach().cpu().numpy()
    corners = [
        (x, y, z)
        for x in (low[0], high[0])
        for y in (low[1], high[1])
        for z in (low[2], high[2])
    ]
    rotated = [_quat_rotate(standing_quat, corner) for corner in corners]
    lows = [min(c[i] for c in rotated) for i in range(3)]
    highs = [max(c[i] for c in rotated) for i in range(3)]
    return lows, highs


def _destroy_quietly(scene) -> None:
    """Best-effort teardown of a throwaway scene (never masks the real error)."""
    try:
        scene.destroy()
    except Exception:  # noqa: BLE001
        pass


class _WorldStub:
    """Minimal duck-typed ``GenesisWorld`` stand-in for ``load_into_scene``."""

    def __init__(self, scene) -> None:
        self.scene = scene


def sanity_check_item(item, *, tolerance: float = 2e-3, render_dir: Optional[Path] = None) -> Dict:
    """Genesis sanity-load one repackaged item (loader + joints + bbox + render).

    Builds a throwaway scene, loads the item through its ``load_into_scene``
    (``gs.morphs.URDF`` with ``merge_fixed_links=False``), asserts a non-empty
    (non-fixed) joint set and cross-checks the descriptor's offline FK union
    bbox against ``entity.get_vAABB()`` at the identity pose (morph
    ``pos=(0, 0, 0)`` with an identity quat ⇒ world == base-link frame).  A
    second throwaway scene renders one frame with the catalog's
    ``standing_quaternion`` for the texture-fidelity eyeball.

    Returns the discovery log (link counts, joint names, bboxes, render path);
    raises on failure — the caller drops the item with a warning.
    """
    import genesis as gs

    local = item.get_local_bounding_box()
    low = local.low.detach().cpu().numpy()
    high = local.high.detach().cpu().numpy()

    scene = gs.Scene(show_viewer=False, renderer=gs.options.renderers.Rasterizer())
    try:
        entity = item.load_into_scene(_WorldStub(scene), pos=(0.0, 0.0, 0.0))
        scene.build(n_envs=0)

        joints = [
            j.name for j in entity.joints if int(j.type) != gs.JOINT_TYPE.FIXED
        ]
        if not joints:
            raise RuntimeError(
                f"no non-fixed joints after load "
                f"(all joints: {[(j.name, int(j.type)) for j in entity.joints]})"
            )
        link_names = [link.name for link in entity.links]
        vaabb = entity.get_vAABB().cpu().numpy()
    finally:
        _destroy_quietly(scene)

    diff_low = abs(vaabb[0] - low).max()
    diff_high = abs(vaabb[1] - high).max()
    if max(diff_low, diff_high) > tolerance:
        raise RuntimeError(
            f"offline FK union bbox does not match entity.get_vAABB() "
            f"(max corner diff {max(diff_low, diff_high):.6f} > {tolerance}): "
            f"offline low={low.tolist()} high={high.tolist()}, "
            f"vAABB low={vaabb[0].tolist()} high={vaabb[1].tolist()}"
        )

    rendered: Optional[str] = None
    if render_dir is not None:
        render_dir.mkdir(parents=True, exist_ok=True)
        render_scene = gs.Scene(show_viewer=False, renderer=gs.options.renderers.Rasterizer())
        try:
            standing = item.standing_quaternion or _PROBE_STANDING_QUAT
            world_low, world_high = _world_aabb_at_standing(item, standing)
            radius = max(max(h - l, 1e-3) for l, h in zip(world_low, world_high)) / 2.0
            # Floor plane for scale/orientation reference; the furniture's lowest
            # world corner is placed on it.
            z_offset = float(-world_low[2])
            center = [
                (l + h) / 2.0 for l, h in zip(world_low, world_high)
            ]
            center[2] += z_offset  # the camera frames the *placed* item
            render_scene.add_entity(gs.morphs.Plane())
            item.load_into_scene(
                _WorldStub(render_scene),
                pos=(0.0, 0.0, z_offset),
                quat=standing,
            )
            # Shallow 3/4 view (mostly horizontal) so tall items stay in frame.
            offset = np.array([1.0, -1.0, 0.6])
            offset = offset / np.linalg.norm(offset) * (2.2 * radius)
            camera = render_scene.add_camera(
                res=(480, 480),
                pos=tuple(center + offset),
                lookat=(center[0], center[1], center[2]),
                fov=50,
            )
            render_scene.build(n_envs=0)
            rgb = camera.render(rgb=True)
            if isinstance(rgb, tuple):
                rgb = rgb[0]
            out_file = render_dir / f"{item.name}_standing.png"
            import imageio

            imageio.imwrite(out_file, np.asarray(rgb)[..., :3])
            rendered = str(out_file)
        finally:
            _destroy_quietly(render_scene)

    return {
        "links": len(link_names),
        "link_names": link_names,
        "moving_joints": joints,
        "local_bbox": (low.tolist(), high.tolist()),
        "vAABB": (vaabb[0].tolist(), vaabb[1].tolist()),
        "bbox_diff": float(max(diff_low, diff_high)),
        "render": rendered,
    }


def run_sanity_gate(
    output_dir: Path,
    item_ids: Sequence[str],
    render_dir: Optional[Path],
) -> Tuple[List[str], Dict[str, Dict]]:
    """Sanity-load every repackaged item; drop (warn on) the failing ones.

    Returns ``(kept_item_ids, reports)``.
    """
    from unienv_genesis_collection import asset_path

    original_assets_dir = asset_path.ASSETS_DIR
    original_download = asset_path.download_assets

    def _no_download(**kwargs):
        raise RuntimeError(
            f"Unexpected Hub download during the sanity gate ({kwargs}); "
            "the repackaged item directory is incomplete."
        )

    # The gate must work from the just-prepared tree only: a Hub download
    # during packaging would mask an incomplete repackage.
    asset_path.download_assets = _no_download
    asset_path.ASSETS_DIR = output_dir.resolve()
    try:
        import genesis as gs
        from genesis.constants import backend as gs_backend

        gs.init(backend=gs_backend.cuda)

        from unienv_genesis_collection.objects.hub_objects.articulated.replicacad_furniture import (
            ReplicaCADFurniture,
        )

        kept: List[str] = []
        reports: Dict[str, Dict] = {}
        for item_id in item_ids:
            item = ReplicaCADFurniture(item_id)
            try:
                report = sanity_check_item(item, render_dir=render_dir)
            except Exception as exc:  # noqa: BLE001 — drop + warn, never fail the run
                print(f"WARNING: dropping '{item_id}' (failed sanity gate): {type(exc).__name__}: {exc}")
                continue
            reports[item_id] = report
            kept.append(item_id)
            print(
                f"OK  {item_id}: links={report['links']} "
                f"moving_joints={len(report['moving_joints'])} "
                f"{report['moving_joints']} "
                f"bbox_diff={report['bbox_diff']:.2e}"
            )
        return kept, reports
    finally:
        asset_path.download_assets = original_download
        asset_path.ASSETS_DIR = original_assets_dir


def prune_catalog(objects_root: Path, dropped_item_ids: Sequence[str]) -> None:
    """Remove the dropped items' entries from ``catalog.json`` (if present)."""
    catalog_file = objects_root / "catalog.json"
    if not catalog_file.is_file() or not dropped_item_ids:
        return
    catalog = json.loads(catalog_file.read_text())
    pruned = {k: v for k, v in catalog.items() if k not in set(dropped_item_ids)}
    catalog_file.write_text(
        json.dumps(pruned, indent=1, ensure_ascii=False) + "\n"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _fetch_dataset() -> Path:
    """Snapshot-download the upstream dataset (cached by huggingface_hub)."""
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            HF_DATASET_REPO,
            repo_type="dataset",
            allow_patterns=[
                f"{URDF_DIR}/**",
                f"{STAGE_DIR}/*.glb",
                f"{ARRANGEMENT_DIR}/*",
                UPSTREAM_LICENSE_FILE,
                "README.md",
            ],
        )
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source", type=Path, default=None,
        help="Existing ReplicaCAD dataset snapshot (with urdf/, stages/, configs/ "
             "and LICENSE.txt). If omitted, the dataset is fetched from "
             f"{HF_DATASET_REPO} into the huggingface_hub cache.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to place the prepared 'objects/replicacad' and "
             "'scenes/rooms/replicacad' trees in.",
    )
    parser.add_argument(
        "--items", type=str, default=None,
        help="Comma-separated subset of item ids to ingest "
             f"(default: all of {', '.join(ARTICULATED_ITEMS)}).",
    )
    parser.add_argument(
        "--skip-gate", action="store_true",
        help="Skip the Genesis GPU sanity-load gate (loader + joints + bbox + render).",
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="Skip the per-item texture render (the gate still checks loads + bboxes).",
    )
    parser.add_argument(
        "--render-dir", type=Path, default=Path("/tmp/replicacad_sanity"),
        help="Directory for the per-item eyeball renders (default /tmp/replicacad_sanity).",
    )
    args = parser.parse_args(argv)

    item_ids = (
        [i.strip() for i in args.items.split(",") if i.strip()]
        if args.items
        else list(ARTICULATED_ITEMS)
    )
    unknown = [i for i in item_ids if i not in ARTICULATED_ITEMS]
    if unknown:
        raise SystemExit(f"Unknown item id(s): {unknown}. Known ids: {list(ARTICULATED_ITEMS)}")

    if args.source is not None:
        source = args.source.expanduser().resolve()
        if not (source / URDF_DIR).is_dir():
            raise SystemExit(f"No 'urdf/' directory in --source: {source}")
    else:
        print(f"Fetching {HF_DATASET_REPO} via huggingface_hub.snapshot_download ...")
        source = _fetch_dataset()
        print(f"  -> {source}")

    objects_root, scenes_root = prepare(source, args.output_dir, item_ids)

    if not args.skip_gate:
        kept, _reports = run_sanity_gate(
            args.output_dir, item_ids, None if args.no_render else args.render_dir
        )
        dropped = sorted(set(item_ids) - set(kept))
        if dropped:
            # Drop the whole item directory + catalog entry of every failure.
            for item_id in dropped:
                shutil.rmtree(objects_root / item_id, ignore_errors=True)
            prune_catalog(objects_root, dropped)
            print(f"Dropped {len(dropped)} item(s): {dropped}")

    print(
        "\nTo upload (requires write access to the org):\n"
        f"  hf upload UniEnvOrg/unienv_genesis_collection_assets \\\n"
        f"      {objects_root} objects/replicacad --repo-type dataset\n"
        f"  hf upload UniEnvOrg/unienv_genesis_collection_assets \\\n"
        f"      {scenes_root} scenes/rooms/replicacad --repo-type dataset"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
