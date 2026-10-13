#!/usr/bin/env python3
# Upstream asset license: Apache-2.0 (per the X-Humanoid/ArtVIP dataset card; the
# upstream snapshot commits no LICENSE file).  This packager only rehosts the USD
# payloads unchanged; the original license applies to the packaged assets.
"""Prepare ArtVIP USD furniture assets for the collection's assets repo.

Downloads (or reuses) the `X-Humanoid/ArtVIP
<https://huggingface.co/datasets/X-Humanoid/ArtVIP>`_ dataset's articulated
objects and repackages them into the layout expected by
``unienv_genesis_collection.objects.hub_objects.articulated.artvip_furniture``
(one self-contained directory per object, loaded with ``gs.morphs.USD`` via
``scene.add_stage``)::

    <output>/objects/artvip/LICENSE                  # attribution (no upstream LICENSE file)
    <output>/objects/artvip/README.md                # provenance + license caveat
    <output>/objects/artvip/catalog.json             # generated annotations + tags
    <output>/objects/artvip/packaging_log.json       # per-object gate log
    <output>/objects/artvip/<object_id>/<object_id>.usd
    <output>/objects/artvip/<object_id>/resource/...# materials, meshes, img/ textures

Upstream layout: ``Articulated_objects/<category>/<subcategory>/<object_id>/model_<id>.usd``
plus a ``resource/`` sidecar directory.  The two **2-level** categories
(``Ikea_furniture``, ``Medical_equipment``) put the product dir directly under
the category (``<category>/<product>/model_*.usd`` + ``resource/``); a
directory *is* the object exactly when it directly contains a ``model_*.usd``,
otherwise it is recursed into, so the sidecar dirs (``resource``, ``origin``,
``.thumbs``, ``NVIDIA``, ``Isaac Sim 4.5``, ``img``, ``v_*``) can never be
packaged as an object.  Packaged object ids flatten the 3-level layout to
``<subcategory>_<object_id>`` (spaces → underscores) and the 2-level one to the
product dir name; the description file is renamed to ``<object_id>.usd``
**unconditionally** (pxr sniffs the layer format by content, so a ``.usda``/
``.usdc`` root renamed to ``.usd`` still opens — the gate/descriptor hard-code
that name).  Stray upstream files Genesis never opens are dropped: OEM
``*.mdl`` materials (the only contents of the ``Isaac Sim 4.5/``
glass-material submodule), ``.thumbs/`` previews, the
``.collect.mapping.json`` metadata and the top-level ``origin`` / ``NVIDIA`` /
``img`` / ``v_*`` sidecar dirs.

Every packaged stage additionally gets four USD-level fixups (A5, A6, C, D):

* **unjointed rigid bodies are deactivated** — Genesis's ``add_stage`` raises
  "Mixed entity detected" for a stage with joints *and* rigid bodies no joint
  references (the loft beds' pillows/blankets);
* **degenerate colliders lose ``CollisionAPI``** — a collision mesh with fewer
  than four non-coplanar points has no 3D hull and ``trimesh``'s convex hull
  aborts on it (``QH6214 ... not enough points(3)`` — the subwaydoor crash);
* **oversized stages are dropped before the load** — a world AABB diagonal
  above ``MAX_STAGE_DIAGONAL_M`` (``--max-size-m``, default 20 m: the two
  shipped ~13.8 m assets gate fine, while subwaycar001's 60 m train overflows
  the SDF int32 shape and machinery001's 79 m allocates >100 GB of voxels).
  The same cheap extent check runs at the head of the gate, so a re-gate of an
  already-packaged oversized object also drops cleanly.

Every packaged object must pass a **Genesis sanity gate**: loader acceptance
through ``ArtVIPFurniture.load_into_scene`` with a single-entity assert, a
non-empty moving (non-fixed) joint set, unique leaf joint names, the offline
USD FK union bbox vs ``entity.get_vAABB()`` at q = 0 *and* at a halfway-opened
configuration (abs/rel tolerances — box-proxy FK vs real mesh geometry),
plus a CUDA-only thumbnail render.  Failing objects are dropped with a
warning instead of failing the run.

License: the ArtVIP dataset card states Apache-2.0 but the snapshot does not
commit a LICENSE file; ``objects/artvip/LICENSE`` carries the attribution
and the provenance README flags the caveat.  All categories (including
``Ikea_furniture`` and ``Medical_equipment``) are packaged under that license;
the original upstream license header is kept at the top of this script.

Usage::

    python scripts/prepare_artvip_assets.py --output-dir /tmp/artvip_assets
    # or with an already-downloaded snapshot:
    python scripts/prepare_artvip_assets.py --source <snapshot-dir> \
        --output-dir /tmp/artvip_assets --categories major_appliances

Then upload to the Hub (needs write access to the org)::

    hf upload UniEnvOrg/unienv_genesis_collection_assets \\
        /tmp/artvip_assets/objects/artvip objects/artvip --repo-type dataset
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Shared packaging/gating library (scripts/_packaging) — the single source
# of truth for the four packagers' gates (see the architecture review §1):
# quat_to_matrix/_WorldStub/_render_thumbnail, write_catalog, the
# asset-completeness check and the sanity gate + its batching all live there.
# Importing the package also installs the genesis_compat shims (idempotent).
# The scripts run as __main__ from the repo root (and the tests load them via
# spec_from_file_location), so scripts/ is not necessarily importable — bind
# it explicitly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
for _path in (str(_REPO_ROOT), str(_SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from _packaging._gate_batch import (  # noqa: E402
    run_sanity_gate_batched as _run_sanity_gate_batched,
)
from _packaging.catalog import write_catalog as _write_catalog_impl  # noqa: E402
from _packaging.gate_common import (  # noqa: E402
    _WorldStub,
    _finite_midpoint,
    _gs_joint_type_name,
    _opened_reference_joints,
    _render_thumbnail,
    quat_to_matrix,
)
from _packaging.sanity import (  # noqa: E402
    BBOX_TOL_ABS,
    BBOX_TOL_REL,
    run_sanity_gate as _run_sanity_gate_impl,
    sanity_check_articulated,
)
from _packaging.usd_verify import verify_object_assets as _verify_object_assets  # noqa: E402

sanity_check_object = sanity_check_articulated  # the pre-refactor family name


HF_DATASET_REPO = "X-Humanoid/ArtVIP"
UPSTREAM_OBJECTS_DIR = "Articulated_objects"
OUTPUT_ASSETS_SUBDIR = "objects/artvip"
HF_UPLOAD_REPO = "UniEnvOrg/unienv_genesis_collection_assets"

# Category exclusions (license caveats are documented in the module docstring
# and README): all categories are now packaged — the upstream license header is
# kept at the top of this script — so the exclusion list is empty.
EXCLUDED_CATEGORIES: Tuple[str, ...] = ()

DEFAULT_CATEGORIES: Tuple[str, ...] = ("major_appliances", "small_furniture")
ALL_KNOWN_CATEGORIES: Tuple[str, ...] = (
    "household_items",
    "industrial_machinery",
    "lab_items",
    "large_furniture",
    "major_appliances",
    "small_appliances",
    "small_furniture",
    # Formerly license-gated; now packaged under the upstream Apache-2.0 card.
    "Ikea_furniture",
    "Medical_equipment",
)
# All known upstream categories are packagable; EXCLUDED_CATEGORIES is empty.
ALL_KNOWN_CATEGORIES += EXCLUDED_CATEGORIES

# Bbox cross-check tolerances (offline-FK box proxy vs real mesh geometry:
# expect deltas up to a few cm where the packaged convex mesh silhouette
# overhangs the link's AABB, or where Genesis subdivides/triangulates)
# — defined once in ``_packaging.sanity`` (BBOX_TOL_ABS/BBOX_TOL_REL) and
# imported above.


# ---------------------------------------------------------------------------
# Category metadata: per-subcategory annotations/tags templates.
#
# ``<subcategory>`` keys; unknown subcategories fall back to the generic
# template.  ``room`` tags are room hints for pool filtering
# (``make_artvip_pool(tags="kitchen", …)``); ``kind`` is the broad annotation.
# Room-hint assignment is a packaging-heuristic, not ground truth —
# ``CATALOG_OVERRIDES`` patches specific ids.
# ---------------------------------------------------------------------------

CATEGORY_TEMPLATES: Dict[str, Dict[str, object]] = {
    # broad annotation      room hint tags
    # --- major/small appliances -------------------------------------------
    "refrigerator": dict(kind="appliance", rooms=("kitchen",), detail="a refrigerator"),
    "dishwasher": dict(kind="appliance", rooms=("kitchen",), detail="a dishwasher"),
    "built-in oven": dict(kind="appliance", rooms=("kitchen",), detail="a built-in oven"),
    "oven": dict(kind="appliance", rooms=("kitchen",), detail="a countertop oven"),
    "gas_cooker": dict(kind="appliance", rooms=("kitchen",), detail="a gas cooker"),
    "large_scale_steamer": dict(kind="appliance", rooms=("kitchen",), detail="a large steamer"),
    "display_chiller": dict(kind="appliance", rooms=("kitchen",), detail="a display chiller cabinet"),
    "range_hood": dict(kind="appliance", rooms=("kitchen",), detail="a range hood"),
    "washing_machine": dict(kind="appliance", rooms=("laundry",), detail="a washing machine"),
    "toilet": dict(kind="fixture", rooms=("bathroom",), detail="a toilet"),
    "water dispenser": dict(kind="appliance", rooms=("utility",), detail="a water dispenser"),
    "ceiling_fan": dict(kind="appliance", rooms=("living_room",), detail="a ceiling fan"),
    "standing fan": dict(kind="appliance", rooms=("living_room",), detail="a standing fan"),
    "standing_air_conditioner": dict(kind="appliance", rooms=("living_room",),
                                     detail="a standing air conditioner"),
    "microwave": dict(kind="appliance", rooms=("kitchen",), detail="a microwave"),
    "airfryer": dict(kind="appliance", rooms=("kitchen",), detail="an air fryer"),
    "coffee machine": dict(kind="appliance", rooms=("kitchen",), detail="a coffee machine"),
    "rice cooker": dict(kind="appliance", rooms=("kitchen",), detail="a rice cooker"),
    "laptop": dict(kind="appliance", rooms=("office",), detail="a laptop"),
    "printer": dict(kind="appliance", rooms=("office",), detail="a printer"),
    "desk_lamp": dict(kind="appliance", rooms=("office",), detail="a desk lamp"),
    "table fan": dict(kind="appliance", rooms=("living_room",), detail="a table fan"),
    "television": dict(kind="appliance", rooms=("living_room",), detail="a television"),
    "cash_register": dict(kind="appliance", rooms=("retail",), detail="a cash register"),
    "sound": dict(kind="appliance", rooms=("living_room",), detail="a sound system"),
    # --- large furniture --------------------------------------------------
    "counter": dict(kind="furniture", rooms=("kitchen",), detail="a counter"),
    "cupboard": dict(kind="furniture", rooms=("kitchen",), detail="a cupboard"),
    "loft_bed_with_desk": dict(kind="furniture", rooms=("bedroom",), detail="a loft bed with desk"),
    "wardrobe": dict(kind="furniture", rooms=("bedroom",), detail="a wardrobe"),
    # --- small furniture ---------------------------------------------------
    "bookcase": dict(kind="furniture", rooms=("living_room",), detail="a bookcase"),
    "cabinet": dict(kind="furniture", rooms=("living_room",), detail="a cabinet"),
    "chair": dict(kind="furniture", rooms=("living_room",), detail="an articulated chair"),
    "door": dict(kind="furniture", rooms=("interior",), detail="an interior door"),
    "easel": dict(kind="furniture", rooms=("studio",), detail="an easel"),
    "faucet": dict(kind="fixture", rooms=("bathroom",), detail="a faucet"),
    "filing_cabinet": dict(kind="furniture", rooms=("office",), detail="a filing cabinet"),
    "floor_lamp": dict(kind="furniture", rooms=("living_room",), detail="a floor lamp"),
    "shoe_cabinet": dict(kind="furniture", rooms=("hallway",), detail="a shoe cabinet"),
    "stool": dict(kind="furniture", rooms=("living_room",), detail="a stool"),
    "table": dict(kind="furniture", rooms=("living_room",), detail="an articulated table"),
    "tea_table": dict(kind="furniture", rooms=("living_room",), detail="a tea table"),
    # --- household items ----------------------------------------------------
    "basket": dict(kind="household item", rooms=("utility",), detail="a basket"),
    "cardboard_box": dict(kind="household item", rooms=("utility",), detail="a cardboard box"),
    "clock": dict(kind="household item", rooms=(), detail="a clock"),
    "holder": dict(kind="household item", rooms=(), detail="a holder"),
    "match": dict(kind="household item", rooms=(), detail="a matchbox"),
    "mop": dict(kind="household item", rooms=("utility",), detail="a mop"),
    "pump_bottle": dict(kind="household item", rooms=("bathroom",), detail="a pump bottle"),
    "showerhead": dict(kind="fixture", rooms=("bathroom",), detail="a showerhead"),
    "stationery": dict(kind="household item", rooms=("office",), detail="a stationery item"),
    "tongs": dict(kind="household item", rooms=("kitchen",), detail="a pair of tongs"),
    "toolbox": dict(kind="household item", rooms=("utility",), detail="a toolbox"),
    "trash_can": dict(kind="household item", rooms=("utility",), detail="a trash can"),
    # --- lab items ------------------------------------------------------------
    "fume_extractor_arm": dict(kind="lab furniture", rooms=("lab",), detail="a fume extractor arm"),
    "glassware": dict(kind="lab item", rooms=("lab",), detail="a glassware item"),
    "lab_bench": dict(kind="lab furniture", rooms=("lab",), detail="a lab bench"),
    "lab_cabinet": dict(kind="lab furniture", rooms=("lab",), detail="a lab cabinet"),
    "microscope": dict(kind="lab item", rooms=("lab",), detail="a microscope"),
    "small_instrument": dict(kind="lab item", rooms=("lab",), detail="a small lab instrument"),
    # --- industrial machinery ---------------------------------------------------
    "agv": dict(kind="machine", rooms=("industrial",), detail="an automated guided vehicle"),
    "cutter": dict(kind="machine", rooms=("industrial",), detail="an industrial cutter"),
    "elevator": dict(kind="machine", rooms=("industrial",), detail="an elevator"),
    "forklift": dict(kind="machine", rooms=("industrial",), detail="a forklift"),
    "machine": dict(kind="machine", rooms=("industrial",), detail="an industrial machine"),
    "security_equipment": dict(kind="machine", rooms=("industrial",), detail="a security scanner"),
    "subway": dict(kind="machine", rooms=("industrial",), detail="a subway vehicle"),
    "ticket_gate": dict(kind="machine", rooms=("industrial",), detail="a ticket gate"),
}

# Generic fallback for a subcategory with no curated template: annotations
# from the syntactic category words, with the top-level kind.
_GENERIC_ROOMS: Dict[str, Tuple[str, ...]] = {
    "major_appliances": ("kitchen",),
    "small_appliances": ("kitchen",),
    "large_furniture": ("living_room",),
    "small_furniture": ("living_room",),
    "household_items": ("utility",),
    "lab_items": ("lab",),
    "industrial_machinery": ("industrial",),
}
_GENERIC_KIND: Dict[str, str] = {
    "major_appliances": "appliance",
    "small_appliances": "appliance",
    "large_furniture": "furniture",
    "small_furniture": "furniture",
    "household_items": "household item",
    "lab_items": "lab item",
    "industrial_machinery": "machine",
}

# Hand fixes (catalog entries) — applied after the generated entries;
# values shape any subset: {"language_annotations": [...], "tags": [...]}.
CATALOG_OVERRIDES: Dict[str, dict] = {}

# Attribution (no upstream LICENSE file is committed in the snapshot —
# the ArtVIP dataset card states Apache-2.0; see the provenance README).
LICENSE_TEXT = """\
ArtVIP articulated object assets.

Dataset: "ArtVIP: Articulated VIP assets" — X-Humanoid/ArtVIP,
https://huggingface.co/datasets/X-Humanoid/ArtVIP

The ArtVIP dataset card declares the Apache-2.0 license
(https://www.apache.org/licenses/LICENSE-2.0). NOTE: the dataset snapshot
does not commit a LICENSE file; the dataset card is the authoritative
license statement — consumers should verify it and attribute
"ArtVIP dataset, © X-Humanoid (Apache-2.0)" when using these models.

All categories (including `Ikea_furniture` and `Medical_equipment`) are
packaged under the upstream Apache-2.0 dataset-card license.

Repackaged by genesis_adaptor/scripts/prepare_artvip_assets.py.
"""

PROVENANCE_README = """\
# ArtVIP articulated USD furniture

Repackaged from the [ArtVIP dataset](https://huggingface.co/datasets/X-Humanoid/ArtVIP)
(articulated USD objects exported from Isaac Sim): one self-contained
directory per object under `objects/artvip/<object_id>/`, containing the
root USD stage (`<object_id>.usd`, renamed from the upstream
`model_<id>.usd`) plus its `resource/` sidecars (material/mesh sublayers and
`resource/img/` textures).  Loaded with `gs.morphs.USD` through
`scene.add_stage` by
`unienv_genesis_collection.objects.hub_objects.articulated.ArtVIPFurniture`.

> **License note.** The ArtVIP dataset card declares **Apache-2.0**, but the
> dataset snapshot does **not commit a LICENSE file** — see `LICENSE` and
> consumers should verify the dataset card before assuming the grant.
> All categories (including `Ikea_furniture` and `Medical_equipment`) are
> packaged under that license.

* Object ids flatten the upstream `Articulated_objects/<category>/<subcategory>/<object_id>/`
  path to `<subcategory>_<object_id>` (spaces → underscores; upstream category
  and id are part of the id only as they appear).
* Upstream `.mdl` OEM materials (`Isaac Sim 4.5/` glass submodule), `.thumbs`
  previews and `.collect.mapping.json` metadata are not shipped — Genesis
  (usd-core) opens neither.
* Genesis grants USD joints by **full prim path**; the descriptor exposes
  *clean*, leaf-prim joint/link names (see the Articulated Rooms docs).
* Every shipped object passed the Genesis sanity gate: single-entity
  `add_stage` load, non-empty moving-joint set, unique joint leaf names, and
  the offline box-proxy FK union bbox cross-checked against
  `entity.get_vAABB()` at the closed *and* a halfway-opened configuration.
  Per-object details are recorded in `packaging_log.json`.

When using these models, please attribute the ArtVIP dataset
— see `LICENSE` and the license note above.
"""


# ---------------------------------------------------------------------------
# Upstream discovery
# ---------------------------------------------------------------------------

def list_artvip_objects(source: Path, categories: Sequence[str]) -> List[Tuple[str, str]]:
    """List ``(packaged_object_id, upstream_dir)`` pairs on *source*.

    A directory *is* an object exactly when it directly contains a
    ``model_*.usd`` root layer; otherwise it is recursed into.  This handles
    both upstream layouts — the common 3-level
    ``<category>/<subcategory>/<object>/`` tree (with its ``resource/``
    sidecar) and the 2-level categories (``Ikea_furniture``,
    ``Medical_equipment``) whose product dir *is* the object dir
    (``model_*.usd`` + ``resource/`` at the same level: recursing into
    ``resource/`` would otherwise package the sidecar as the object and
    never the real asset).

    The object id flattens ``Articulated_objects/<category>/<subcategory>/<id>``
    to ``<subcategory>_<id>`` (spaces → underscores), collapsing the doubled
    prefix when the upstream id already starts with the subcategory token
    (``cabinet/cabinet_12`` → ``cabinet_12``, ``refrigerator/refrigerator002``
    → ``refrigerator002``).  Duplicated flattened ids (a packaging bug) abort
    the run; categories absent from a (possibly partial) snapshot are
    skipped with a note.
    """
    found: Dict[str, Path] = {}
    for category in categories:
        cat_dir = source / UPSTREAM_OBJECTS_DIR / category
        if not cat_dir.is_dir():
            print(
                f"NOTE: category {category!r} is not in the source snapshot "
                f"({source}); skipping."
            )
            continue
        for object_dir in _walk_object_dirs(cat_dir):
            if object_dir.parent == cat_dir:
                # 2-level category (Ikea_furniture / Medical_equipment): there
                # is no subcategory level, so the product dir *is* the id.
                flat_id = object_dir.name.replace(" ", "_")
            else:
                # The flattened id uses the *upstream parent* dir of the
                # object (the subcategory level of the 3-level layout) as its
                # prefix.
                sub_name = object_dir.parent.name.replace(" ", "_")
                sub_flat = sub_name.lower()
                upstream_flat = object_dir.name.replace(" ", "_").lower()
                if upstream_flat.startswith(sub_flat):
                    flat_id = object_dir.name.replace(" ", "_")
                else:
                    flat_id = f"{sub_name}_{upstream_flat}"
            if flat_id in found:
                raise SystemExit(
                    f"Duplicate packaged object id {flat_id!r} "
                    f"(already mapped to {found[flat_id]}; new={object_dir}). "
                    "See the module docstring on id flattening."
                )
            found[flat_id] = object_dir
    return sorted(found.items())


# Upstream directory names that are sidecars/never objects (recursed past or
# skipped): ``resource/`` carries the material/mesh sublayers + textures,
# ``origin/`` the original authoring file, ``.thumbs`` the previews, and the
# ``NVIDIA`` / ``Isaac Sim 4.5`` / ``img`` / ``v_*`` dirs hold preview images
# and OEM material packages — none of them is an articulated object.
_SIDECAR_DIR_NAMES = frozenset({
    "resource", "origin", ".thumbs", "NVIDIA", "Isaac Sim 4.5", "img",
})


def _is_object_dir(path: Path) -> bool:
    """A directory *is* the object when it directly holds a ``model_*.usd``."""
    return any(
        p.is_file() and p.stem.startswith("model_")
        for p in path.iterdir()
        if p.suffix.lower() in (".usd", ".usda", ".usdc", ".usdz")
    )


def _is_sidecar_entry(name: str) -> bool:
    """A *top-level* entry that is never part of a stage's closure.

    ``resource/`` is excluded: it holds the material/mesh sublayers and
    textures the packaged stage references (verified over the whole snapshot
    — ``resource/`` is the only top-level dir ever in ``_usd_required_files``).
    """
    return name.startswith("v_") or (
        name in _SIDECAR_DIR_NAMES and name != "resource"
    )


def _walk_object_dirs(root: Path):
    """Yield object dirs under *root*, depth-first, sorted.

    recursion terminates at an object dir (its subtree is the asset), and
    sidecar dir names are never entered.
    """
    for entry in sorted(p for p in root.iterdir() if p.is_dir()):
        name = entry.name
        if name.startswith("v_") or name in _SIDECAR_DIR_NAMES:
            continue
        if _is_object_dir(entry):
            yield entry
            continue
        yield from _walk_object_dirs(entry)


# ---------------------------------------------------------------------------
# Repackaging
# ---------------------------------------------------------------------------

# Upstream sidecars that never enter the packaged tree.
_DROPPED_SUFFIXES = (".mdl",)
_DROPPED_PATH_PARTS = (".thumbs", ".collect.mapping.json", ".gitattributes")
# `Isaac Sim 4.5/` is the OEM glass-material submodule — its only payloads are
# `*.mdl` files, so it collapses to empty once files are filtered.
_SKIPPED_TOP_DIRS = ("Isaac Sim 4.5",)


def _find_model_usd(obj_dir: Path) -> Optional[Path]:
    """The upstream stage's root ``model_*.usd`` layer (exactly one expected).

    The stem must start with ``model_`` so stray non-root layers (e.g.
    ``material.usd``) can never be picked as the stage root; when several
    ``model_*`` layers exist the choice is ambiguous (``None`` → dropped).
    """
    candidates = sorted(
        p for p in obj_dir.glob("model_*")
        if p.is_file() and p.suffix.lower() in (".usd", ".usda", ".usdc", ".usdz")
    )
    return candidates[0] if len(candidates) == 1 else None


def repackage_object(obj_dir: Path, object_id: str, out_root: Path) -> Path:
    """Repackage one upstream object dir into ``out_root/<object_id>/``.

    Copies the tree verbatim (minus the dropped sidecars above), renames
    the root layer ``model_<id>.usd`` → ``<object_id>.usd`` and verifies
    that the pxr layer stack + ``resource/**`` textures resolve inside the
    packaged directory (fail → raises; the caller drops the object).
    """
    model_usd = _find_model_usd(obj_dir)
    if model_usd is None:
        raise FileNotFoundError(f"No single 'model_*.usd' root layer in {obj_dir}.")

    obj_out_dir = out_root / object_id
    obj_out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(obj_dir.rglob("*")):
        rel = path.relative_to(obj_dir)
        if rel.parts[0] in _SKIPPED_TOP_DIRS:
            continue
        if any(part in _DROPPED_PATH_PARTS for part in rel.parts):
            continue
        # Top-level sidecars (``origin/``, ``NVIDIA/``, ``img/``, ``v_*``) are
        # previews / original authoring files — never referenced by a stage.
        if _is_sidecar_entry(rel.parts[0]):
            continue
        if path.suffix.lower() in _DROPPED_SUFFIXES:
            continue
        if path.is_dir():
            (obj_out_dir / rel).mkdir(parents=True, exist_ok=True)
            continue
        (obj_out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, obj_out_dir / rel)
        copied += 1

    # Rename the root layer to the packaged <object_id>.usd convention —
    # UNCONDITIONALLY ``.usd``, even when the upstream root is a ``.usda``/
    # ``.usdc``: the gate/descriptor contract hard-codes ``<object_id>.usd``
    # and Usd.Stage.Open sniffs the layer format by *content*, so a renamed
    # binary/Crate layer still opens fine (a suffix-preserving rename would
    # leave the gate facing ``<id>.usda`` and pxr raises "Failed to open
    # layer" on the ``<id>.usd`` it expects).
    packaged_model = obj_out_dir / model_usd.relative_to(obj_dir)
    renamed = obj_out_dir / f"{object_id}.usd"
    packaged_model.rename(renamed)
    # prune directories left empty by the drops
    for path in sorted((str(p) for p in obj_out_dir.rglob("*") if p.is_dir()),
                       key=lambda item: -len(item)):
        try:
            Path(path).rmdir()
        except OSError:
            pass
    _fixup_packaged_stage(renamed, object_id)
    _verify_object_assets(renamed, obj_out_dir)
    # Oversized-asset guard (A7): a stage whose world AABB diagonal exceeds
    # MAX_STAGE_DIAGONAL_M (tens-of-metres trains/machinery) blows up the
    # runtime load far past anything furniture-shaped — the SDF voxelisation
    # allocates 10s of GB (machinery001: 79 m → 129 GB) or overflows the SDF
    # int32 shape (subwaycar001: 60 m train).  Dropped here, *before* the
    # gate ever loads it, with the measured extent in the log line.
    diagonal = _stage_diagonal(renamed)
    if diagonal > MAX_STAGE_DIAGONAL_M:
        raise OversizedAssetError(
            f"stage world AABB diagonal {diagonal:.2f} m exceeds the "
            f"{MAX_STAGE_DIAGONAL_M:.1f} m packaging limit — dropped before "
            "the (very expensive) Genesis load"
        )
    # Copy file tally lands in the gate record via the object dir scan.
    return obj_out_dir


# Oversized-asset guard threshold.  Two shipped ArtVIP assets are genuinely
# ~13.8 m long (``security_equipment_security_check_equipment``, ``machine006``)
# and gate fine, so the cut sits above them: it exists to reject the
# tens-of-metres outliers (``subwaycar001`` 60 m, ``machinery001`` 79 m) whose
# Genesis load either overflows the SDF int32 shape or allocates >100 GB.
MAX_STAGE_DIAGONAL_M = 20.0


class OversizedAssetError(RuntimeError):
    """The stage's world extent is far beyond any furniture scale (A7 guard)."""


def _stage_diagonal(usd_file: Path) -> float:
    """Cheap world-AABB diagonal of a stage (pxr ``BBoxCache``, stage units)."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_file))
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=True
    )
    lows: List[np.ndarray] = []
    highs: List[np.ndarray] = []
    for child in stage.GetPseudoRoot().GetChildren():
        aligned = cache.ComputeWorldBound(child).ComputeAlignedRange()
        if aligned.IsEmpty():
            continue
        lows.append(np.asarray(aligned.GetMin(), dtype=float))
        highs.append(np.asarray(aligned.GetMax(), dtype=float))
    if not lows:
        return 0.0
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    return float(
        np.linalg.norm(np.max(highs, axis=0) - np.min(lows, axis=0)) * meters_per_unit
    )


def _fixup_packaged_stage(usd_file: Path, object_id: str) -> dict:
    """USD-level repackaging fixups on the packaged root layer (A5 + A6).

    * **A5 — unjointed rigid bodies.**  Genesis's ``add_stage`` raises
      "Mixed entity detected" for a stage that has joints *and* rigid bodies
      no joint references (ArtVIP's loft beds ship pillows/blankets as loose
      ``RigidBodyAPI`` prims inside the articulation subtree).  Such prims
      (and their subtrees) are **deactivated** — a strong opinion in the root
      layer that hides the composed prim from every consumer (pxr's default
      ``Traverse``/``PrimRange`` predicates skip inactive prims), which
      removing the root-layer spec alone could not do because the prim is
      also defined by a referenced link layer.
    * **A6 — degenerate colliders.**  A collision mesh with fewer than four
      non-coplanar points has no 3D hull; ``trimesh.convex.convex_hull``
      aborts on it (``QH6214 ... not enough points(3)`` — the subwaydoor
      gate crash).  ``UsdPhysics.CollisionAPI`` is removed from those meshes
      (the visual role, if any, is untouched).
    * **C — inverted collider winding (RC-3).**  A *closed* collision mesh
      wound inside-out makes Genesis's SDF wall-thickness refinement fail:
      the probe casts rays against each face normal from just off the
      surface, and on an inverted closed shell the normals point outward,
      so every ray escapes and ``genesis.utils.mesh.get_wall_thickness``
      indexes an empty hit array (``IndexError: index -1 is out of bounds
      for axis 0 with size 0`` — microwave_2's
      ``/root/E_base_plate_1/P_d713ed04dd5c91``).  The predicate reproduces
      that exact failure offline (watertight after Genesis's vertex weld +
      consistent winding + negative signed volume + a **zero-hit probe**)
      and only then reverses ``faceVertexIndices`` — a merely negative-
      volume mesh whose rays still hit (locally flipped patches —
      refrigerator_fridge, the tea_table drawers) probes fine and is left
      alone, as are the 40+ non-watertight/convexified negative-volume
      colliders that load fine; non-triangle meshes are skipped (winding
      judgement needs a triangulation).
    * **D — world-welded orphan joints (RC-4).**  A joint with an EMPTY
      ``body0`` relationship is welded to the *world* in USD's semantics,
      so Genesis roots it as its own entity and the stage splits in two
      (``ArticulationError``: "split into 2 entities") whenever the body1
      side is a non-root rigid body — oven_1's door hinge
      (``/root/E_door_197/RevoluteJoint``) and table_dressing_table004's
      mirror weld (``/root/E_mirror_35/FixedJoint``).  Such joints are
      **re-rooted onto the main articulation root** (the rigid body that is
      no joint's ``body1``): ``Body0Rel`` is re-pointed and
      ``localPos0``/``localRot0`` re-expressed in that root's frame so the
      joint frame keeps its exact authored world placement.  This is a
      **structural edit matching the authored intent**: the door/mirror
      become children of the main body (they already hang in its subtree
      spatially) instead of being world-anchored strays.  A legitimate
      world-anchor on the root body itself (``body1`` == the main root) is
      left untouched.

    Returns the fixup tally (logged by the caller).
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(usd_file))
    if stage is None:
        raise FileNotFoundError(f"Cannot open the packaged stage {usd_file}.")

    # ---- A5: prune rigid bodies no joint references ------------------------
    joint_targets: List[str] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        for rel in (joint.GetBody0Rel(), joint.GetBody1Rel()):
            joint_targets.extend(str(target) for target in rel.GetTargets())

    def _referenced(path: str) -> bool:
        # A joint's target is the body prim itself or a descendant of it
        # (Genesis resolves the nearest enclosing RigidBodyAPI prim).
        return any(t == path or t.startswith(path + "/") for t in joint_targets)

    unjointed = [
        prim
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and not _referenced(str(prim.GetPath()))
    ]
    pruned_paths: List[str] = []
    for prim in unjointed:
        if not prim.IsValid():
            continue
        pruned_paths.append(str(prim.GetPath()))
        prim.SetActive(False)
    if pruned_paths:
        print(
            f"  [repackage] {object_id}: pruned {len(pruned_paths)} unjointed "
            f"RigidBodyAPI prim(s) — {', '.join(pruned_paths[:4])}"
            + (" ..." if len(pruned_paths) > 4 else "")
        )

    # ---- A6: prune degenerate collision meshes -----------------------------
    degenerate: List[str] = []
    for prim in stage.Traverse():
        if not (prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI)):
            continue
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        array = np.asarray(points, dtype=float).reshape(-1, 3) if points else np.zeros((0, 3))
        if _is_degenerate_collider(array):
            degenerate.append(str(prim.GetPath()))
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
    if degenerate:
        print(
            f"  [repackage] {object_id}: pruned {len(degenerate)} degenerate "
            f"collision mesh(es) (<4 non-coplanar points) — "
            f"{', '.join(degenerate[:4])}" + (" ..." if len(degenerate) > 4 else "")
        )

    # ---- C: flip inverted (watertight, negative-volume) colliders ----------
    # The predicate is the exact runtime failure, reproduced offline:
    #   * Genesis welds coincident vertices of non-convex collision meshes
    #     (``merge_vertices``) before the SDF stage, so watertightness is
    #     judged on the WELDED mesh;
    #   * the wall-thickness probe (``genesis/utils/mesh.py:get_wall_thickness``)
    #     samples face centres, steps JUST OFF the surface against each face
    #     normal and casts a ray back along −normal; on an inside-out closed
    #     shell the normals point outward, every ray escapes, and the empty
    #     hit array is indexed (``IndexError: index -1 is out of bounds for
    #     axis 0 with size 0``) — microwave_2's
    #     ``/root/E_base_plate_1/P_d713ed04dd5c91``;
    #   * a merely *negative-volume* mesh whose rays still hit (locally
    #     flipped patches — refrigerator_fridge / tea_table* drawers) probes
    #     fine and must NOT be flipped; nor may non-triangle or non-welded-
    #     watertight meshes (the 40+ benign negative-volume colliders that
    #     convexify or stay open at runtime).
    # Reversing ``faceVertexIndices`` (and nothing else) turns the shell
    # right-side-out; the vertex positions and the AABB are untouched.
    inverted: List[str] = []
    for prim in stage.Traverse():
        if not (prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI)):
            continue
        mesh = UsdGeom.Mesh(prim)
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        points = mesh.GetPointsAttr().Get()
        if not counts or not indices or not points:
            continue
        counts = np.asarray(counts, dtype=np.int64)
        if (counts != 3).any():
            continue  # winding judgement needs a triangulation — skip conservatively
        faces = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
        if len(faces) == 0:
            continue
        import trimesh

        welded = trimesh.Trimesh(
            vertices=np.asarray(points, dtype=np.float64).reshape(-1, 3),
            faces=faces,
            process=False,
        )
        welded.merge_vertices()
        if (
            not welded.is_watertight
            or not welded.is_winding_consistent
            or not np.isfinite(welded.volume)
            or welded.volume >= 0.0
        ):
            continue
        # The failure itself: the probe's rays all escape (zero hits).
        diagonal = np.linalg.norm(
            welded.vertices.max(axis=0) - welded.vertices.min(axis=0)
        )
        stride = max(1, len(welded.faces) // 1000)
        centres = welded.triangles_center[::stride]
        normals = welded.face_normals[::stride]
        origins = centres - 1e-3 * diagonal * normals
        _, ray_indices, _ = welded.ray.intersects_location(
            origins, -normals, multiple_hits=False
        )
        if len(ray_indices) != 0:
            continue  # rays still hit — probes fine at runtime, leave it
        mesh.GetFaceVertexIndicesAttr().Set(
            [int(v) for v in faces[:, ::-1].reshape(-1)]
        )
        inverted.append(str(prim.GetPath()))
    if inverted:
        print(
            f"  [repackage] {object_id}: flipped {len(inverted)} inverted "
            f"collider(s) (watertight, negative signed volume) — "
            f"{', '.join(inverted[:4])}" + (" ..." if len(inverted) > 4 else "")
        )

    # ---- D: re-root world-welded orphan joints -----------------------------
    rerooted = _reparent_world_welded_joints(stage, object_id)

    if pruned_paths or degenerate or inverted or rerooted:
        stage.GetRootLayer().Save()
    return dict(
        unjointed_bodies=pruned_paths,
        degenerate_colliders=degenerate,
        inverted_colliders=inverted,
        rerooted_joints=rerooted,
    )


def _reparent_world_welded_joints(stage, object_id: str) -> List[str]:
    """D (RC-4): re-root joints welded to the WORLD onto the main root.

    A joint with an EMPTY ``body0`` relationship is welded to the world in
    USD's semantics: its ``localPos0``/``localRot0`` are world-frame
    quantities (debugger-verified on oven_1: ``localPos0`` is the child's
    own world position).  Genesis roots such a joint as its own entity, so
    a stage whose ``body1`` side is a *non-root* rigid body splits in two —
    ``ArticulationError``: "split into 2 entities" (oven_1's door hinge
    ``/root/E_door_197/RevoluteJoint``, table_dressing_table004's mirror
    weld ``/root/E_mirror_35/FixedJoint``).

    The main articulation root is the rigid body that is **no joint's
    ``body1``** (the offline FK walker's root rule).  When exactly one such
    body exists, every world-welded joint whose ``body1`` resolves to a
    *different* body is re-rooted onto it: ``Body0Rel`` is re-pointed and
    ``localPos0``/``localRot0`` re-expressed so the joint frame keeps its
    exact authored world placement — translation through the root's full
    authored world matrix, rotation through its rigid (polar-decomposed)
    part.  This is a **structural edit matching the authored intent**: the
    door/mirror become children of the main body they already hang beside,
    instead of world-anchored strays.  A legitimate world-anchor on the
    root body itself (``body1`` == the main root) is left untouched, as is
    any stage whose root determination is ambiguous (0 or ≥2 candidates).

    Returns the re-rooted joint paths.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    def world(prim) -> np.ndarray:
        matrix = xform_cache.GetLocalToWorldTransform(prim)
        return np.array(
            [[matrix[i][j] for j in range(4)] for i in range(4)], dtype=np.float64
        ).T

    def resolve(path_str):
        if not path_str:
            return None
        walked = stage.GetPrimAtPath(path_str)
        while walked.IsValid() and str(walked.GetPath()) != "/":
            if walked.HasAPI(UsdPhysics.RigidBodyAPI):
                return walked
            walked = walked.GetParent()
        return None

    joint_prims = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)]

    def body1_of(prim):
        targets = UsdPhysics.Joint(prim).GetBody1Rel().GetTargets()
        return resolve(str(targets[0]) if targets else None)

    # Bodies with a REAL parent (a joint whose body0 is set).  The remaining
    # bodies are unjointed or welded to the world by a body0-less joint —
    # those are the root candidates (the offline FK walker's root rule).
    parented = set()
    for prim in joint_prims:
        body0_targets = UsdPhysics.Joint(prim).GetBody0Rel().GetTargets()
        if not body0_targets:
            continue  # world-welded — says nothing about the body1's parentage
        body1 = body1_of(prim)
        if body1 is not None:
            parented.add(str(body1.GetPath()))
    candidates = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        and str(prim.GetPath()) not in parented
    ]
    if len(candidates) > 1:
        # Several bodies lack a real parent (a main root AND an RC-4 orphan,
        # or a genuinely multi-root stage): disambiguate by the body0 role —
        # the main articulation root is the candidate other bodies' joints
        # parent onto (resolved to the nearest RigidBodyAPI prim).  Still
        # ambiguous ⇒ not the RC-4 shape; untouched.
        parents_of = set()
        for prim in joint_prims:
            joint = UsdPhysics.Joint(prim)
            body0_targets = joint.GetBody0Rel().GetTargets()
            if not body0_targets:
                continue
            body0 = resolve(str(body0_targets[0]))
            body1 = body1_of(prim)
            if body0 is None or body1 is None or body0 == body1:
                continue
            parents_of.add(str(body0.GetPath()))
        candidates = [path for path in candidates if path in parents_of]
    if len(candidates) != 1:
        # No single main root (0 candidates, a world-anchored root + several
        # siblings, …): not the RC-4 shape — untouched.
        return []
    root_path = candidates[0]
    root_world = world(stage.GetPrimAtPath(root_path))
    root_world_inv = np.linalg.inv(root_world)
    # Rigid rotation of the root frame (the closest rotation — the polar
    # decomposition's orthogonal factor, reflection folded away; exact when
    # the root frame is rigid, the common case).
    U, _, Vt = np.linalg.svd(root_world[:3, :3])
    root_R = U @ Vt
    if np.linalg.det(root_R) < 0.0:
        U[:, -1] = -U[:, -1]
        root_R = U @ Vt

    rerooted: List[str] = []
    for prim in joint_prims:
        joint = UsdPhysics.Joint(prim)
        body0_targets = joint.GetBody0Rel().GetTargets()
        if body0_targets:
            continue  # genuinely parented — untouched
        body1_targets = joint.GetBody1Rel().GetTargets()
        body1 = resolve(str(body1_targets[0]) if body1_targets else None)
        if body1 is None or str(body1.GetPath()) == root_path:
            continue  # root bodies may stay world-anchored
        pos0 = joint.GetLocalPos0Attr().Get() if joint.GetLocalPos0Attr().HasValue() else None
        rot0 = joint.GetLocalRot0Attr().Get() if joint.GetLocalRot0Attr().HasValue() else None
        p_world = (
            np.asarray(pos0, dtype=np.float64).reshape(3)
            if pos0 is not None
            else np.zeros(3)
        )
        R_world = (
            _usd_quat_to_rotation_matrix(rot0) if rot0 is not None else np.eye(3)
        )
        # Keep the joint frame's world placement: express it relative to the
        # root body's frame instead of the world's.
        p_new = (root_world_inv @ np.array([*p_world, 1.0]))[:3]
        R_new = root_R.T @ R_world
        joint.GetBody0Rel().SetTargets([Sdf.Path(root_path)])
        pos_attr = joint.GetLocalPos0Attr()
        if not pos_attr:
            pos_attr = joint.CreateLocalPos0Attr()
        if str(pos_attr.GetTypeName()).endswith("d"):
            pos_attr.Set(Gf.Vec3d(*(float(v) for v in p_new)))
        else:
            pos_attr.Set(Gf.Vec3f(*(float(v) for v in p_new)))
        rot_attr = joint.GetLocalRot0Attr()
        if not rot_attr:
            rot_attr = joint.CreateLocalRot0Attr()
        quat = _rotation_matrix_to_usd_quat(R_new)
        if str(rot_attr.GetTypeName()).endswith("d"):
            rot_attr.Set(
                Gf.Quatd(
                    quat[0], Gf.Vec3d(*(float(v) for v in quat[1:]))
                )
            )
        else:
            rot_attr.Set(Gf.Quatf(quat[0], Gf.Vec3f(*(float(v) for v in quat[1:]))))
        rerooted.append(str(prim.GetPath()))
    if rerooted:
        print(
            f"  [repackage] {object_id}: re-rooted {len(rerooted)} world-welded "
            f"joint(s) onto the articulation root {root_path} — "
            f"{', '.join(rerooted)}"
        )
    return rerooted


def _usd_quat_to_rotation_matrix(quat) -> np.ndarray:
    """pxr quaternion (``.real`` + ``.imaginary`` xyz) → ``(3, 3)`` rotation."""
    qi = np.asarray(quat.imaginary, dtype=np.float64)
    w = float(quat.real)
    x, y, z = qi
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _rotation_matrix_to_usd_quat(matrix: np.ndarray) -> Tuple[float, float, float, float]:
    """``(3, 3)`` rotation → normalised ``(w, x, y, z)`` quaternion tuple
    (Shepperd's method — pxr's ``Gf.Rotation`` build lacks a matrix ctor)."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        quat = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        quat = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        quat = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    quat /= np.linalg.norm(quat)
    return tuple(float(v) for v in quat)


def _is_degenerate_collider(points: np.ndarray, rel_tol: float = 1e-9) -> bool:
    """A point set with no four non-coplanar points (no 3D convex hull)."""
    if len(points) < 4:
        return True
    if not np.isfinite(points).all():
        return True
    centred = points - points.mean(axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    if not np.isfinite(singular).all():
        return True
    threshold = max(1e-12, float(singular[0]) * rel_tol)
    return int((singular > threshold).sum()) < 3



# ---------------------------------------------------------------------------
# Generated catalog
# ---------------------------------------------------------------------------

def write_catalog(
    out_root: Path,
    records: Sequence[dict],
    merge: bool = True,
    drop: Optional[Sequence[str]] = None,
) -> int:
    """Write ``catalog.json`` from the gate records (+ ``CATALOG_OVERRIDES``).

    Thin family wrapper over :func:`_packaging.catalog.write_catalog` (the
    single implementation): entries ``{"language_annotations": [...],
    "tags": [...]}`` (the shared ``load_catalog`` schema); incremental runs
    (``--objects`` subsets into an already-populated output tree) keep the
    entries of objects that are *not* part of this run — a re-run of the same
    object rewrites its entry (and its files), so a stale entry cannot
    outlive its packaged tree.
    """
    return _write_catalog_impl(
        out_root, records, merge=merge, drop=drop, overrides=CATALOG_OVERRIDES,
    )

def annotate_object(
    upstream_dir: Path,
    n_revolute: int,
    n_prismatic: int,
) -> Tuple[str, str, dict]:
    """One catalog entry from the category templates + the gate's joint counts.

    Annotations broad → detailed (ordering contract of
    :func:`~unienv_genesis_collection.objects.hub_objects.catalog.load_catalog`):
    ``[kind, "a <subcategory>", "<the joint-summary line>"]`` — the detailed
    line carries the gate-observed DOF kinds (nothing per-asset beyond that
    is baked).  Returns ``(object_kind, detail_line, entry)``; room-hint tags
    come from the template's ``rooms`` heuristic and the ``articulated``
    tag is added by the caller (the only catalogue-level articulation
    signal, like the RoboCasa pipeline's ``n_joints``).

    ``CATALOG_OVERRIDES`` patches specific ids afterwards (see
    :func:`write_catalog`).
    """
    parts = upstream_dir.relative_to(upstream_dir.parents[2]).parts  # (cat, sub, id)
    category, subcategory = parts[0], parts[1]
    template = CATEGORY_TEMPLATES.get(subcategory)
    if template is None:
        template = dict(
            kind=_GENERIC_KIND.get(category, "object"),
            rooms=_GENERIC_ROOMS.get(category, ()),
        )
    subcat_words = _subcategory_words(subcategory)
    kind = str(template["kind"])
    annotations = [kind, f"a {subcat_words}"]
    # Detailed line: from the subjacent gate counts (post-gate), canonical
    # template word when the counts are not yet available.
    specifics = []
    if n_revolute:
        if _is_door_like(subcategory):
            specifics.append(f"{n_revolute} hinged door{'' if n_revolute == 1 else 's'}")
        else:
            specifics.append(f"{n_revolute} hinged part{'' if n_revolute == 1 else 's'}")
    if n_prismatic:
        specifics.append(
            f"{n_prismatic} drawer{'' if n_prismatic == 1 else 's'}"
            if subcategory in ("filing_cabinet", "shoe_cabinet")
            else f"{n_prismatic} sliding part{'' if n_prismatic == 1 else 's'}"
        )
    annotations.append(
        f"a {subcat_words} with {', '.join(specifics)}"
        if specifics
        else f"a plain {subcat_words} variant"
    )
    entry_tags = {kind.replace(" ", "_"), subcategory.replace(" ", "_")}
    entry_tags.update(tuple(template["rooms"]))  # type: ignore[arg-type]
    entry = {
        "language_annotations": list(dict.fromkeys(annotations)),
        "tags": sorted(entry_tags),
    }
    return kind, subcat_words, entry


def _subcategory_words(subcategory: str) -> str:
    return subcategory.replace("_", " ")


def _is_door_like(subcategory: str) -> bool:
    words = _subcategory_words(subcategory)
    return any(w in words for w in ("door", "fridge", "refrigerator", "cupboard",
                                    "wardrobe", "oven", "microwave", "chiller",
                                    "steam", "cooker", "dishwasher", "washer",
                                    "dryer", "tray", "booth", "airfryer"))


# ---------------------------------------------------------------------------
# Genesis sanity gate (GPU)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Genesis sanity gate (GPU) — the shared loop (``_packaging.sanity``) wired to
# this family's descriptor factory + the A7 oversized-asset pre-check
# ---------------------------------------------------------------------------

def _gate_descriptor(object_id: str):
    """Build the family descriptor for one packaged id (the only per-family
    part of the gate).  Imported lazily so the asset repointing
    (``asset_path.ASSETS_DIR`` → *output_dir*) is already in place."""
    from unienv_genesis_collection.objects.hub_objects.articulated.artvip_furniture import (
        ArtVIPFurniture,
    )

    return ArtVIPFurniture(object_id)


def _gate_pre_check(object_id: str, item, max_size_m: float):
    """Oversized-asset guard (A7): cheap pxr extent check BEFORE the
    (potentially 100+ GB) Genesis load, so an over-scale asset is dropped
    cleanly with a logged reason instead of crashing the worker (SDF int32
    overflow / OOM-kill).  Returns the drop record, or ``None`` to proceed."""
    try:
        diagonal = _stage_diagonal(item.description_file)
    except Exception as exc:  # noqa: BLE001 — the loader reports it
        diagonal = 0.0
        print(
            f"  [gate] {object_id}: extent pre-check failed "
            f"({type(exc).__name__}: {exc}) — falling through to the load."
        )
    if diagonal > max_size_m:
        print(
            f"  [gate] {object_id}: DROPPED — oversized "
            f"(world AABB diagonal {diagonal:.2f} m > {max_size_m:.1f} m); "
            "not loaded."
        )
        return dict(
            object_id=object_id, status="dropped",
            error=f"OversizedAsset: stage world AABB diagonal "
                  f"{diagonal:.2f} m exceeds the {max_size_m:.1f} m limit",
        )
    return None


def run_sanity_gate(
    output_dir: Path,
    object_ids: Sequence[str],
    render_dir: Optional[Path] = None,
    bbox_tol_abs: float = BBOX_TOL_ABS,
    bbox_tol_rel: float = BBOX_TOL_REL,
    max_size_m: Optional[float] = None,
) -> Tuple[List[str], List[dict], List[dict]]:
    """Sanity-load every packaged object; drop (warn on) the failing ones.

    Runs from the freshly packaged tree only (a Hub download during
    packaging would mask an incomplete repackage — assets are repointed at
    *output_dir* and downloads are stubbed out).  Every object's stage world
    extent is checked against *max_size_m* before the Genesis load (the A7
    oversized-asset guard).  Returns
    ``(kept_object_ids, passed_records, dropped_records)``.
    """
    # The gate worker subprocess is a plain ``run_sanity_gate`` call (the
    # batching helper forwards no extra kwargs), so the size limit travels
    # through the environment; the CLI's --max-size-m sets it in prepare().
    if max_size_m is None:
        max_size_m = float(
            os.environ.get("ARTVIP_MAX_SIZE_M", str(MAX_STAGE_DIAGONAL_M))
        )

    def _pre_check(object_id: str, item):
        return _gate_pre_check(object_id, item, max_size_m)

    return _run_sanity_gate_impl(
        output_dir, object_ids,
        descriptor_factory=_gate_descriptor,
        check=sanity_check_object,
        render_dir=render_dir,
        bbox_tol_abs=bbox_tol_abs,
        bbox_tol_rel=bbox_tol_rel,
        pre_check=_pre_check,
    )

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def prepare(
    source: Path,
    output_dir: Path,
    categories: Sequence[str],
    objects: Optional[Sequence[str]] = None,
    gate: bool = True,
    bbox_tol_abs: float = BBOX_TOL_ABS,
    bbox_tol_rel: float = BBOX_TOL_REL,
    render_dir: Optional[Path] = None,
    max_size_m: float = MAX_STAGE_DIAGONAL_M,
) -> Tuple[Path, List[str]]:
    """Package the selected categories into ``output_dir/objects/artvip``.

    Returns ``(out_root, kept_object_ids)``.
    """
    out_root = output_dir / OUTPUT_ASSETS_SUBDIR
    out_root.mkdir(parents=True, exist_ok=True)

    pairs = list_artvip_objects(source, categories)
    if objects:
        wanted = {o.strip().lower() for o in objects if o.strip()}
        # Packaged ids and upstream ids, exact or substring.
        selected = [
            (object_id, obj_dir)
            for object_id, obj_dir in pairs
            if (object_id.lower() in wanted
                or obj_dir.name.lower() in wanted
                or any(w in object_id.lower() or w in obj_dir.name.lower() for w in wanted))
        ]
        if not selected:
            raise SystemExit(
                f"--objects matched none of the {len(pairs)} upstream objects."
            )
    else:
        selected = pairs
    if not selected:
        raise SystemExit(f"No ArtVIP objects found in {source} for {list(categories)}.")

    upstream_dirs: Dict[str, Path] = {}
    packaged: List[str] = []
    print(f"Repackaging {len(selected)} object(s) ...")
    for object_id, obj_dir in selected:
        try:
            repackage_object(obj_dir, object_id, out_root)
        except Exception as exc:  # noqa: BLE001 — repackage failure ⇒ drop
            print(f"  [repackage] {object_id}: DROPPED — {type(exc).__name__}: {exc}")
            shutil.rmtree(out_root / object_id, ignore_errors=True)
            continue
        upstream_dirs[object_id] = obj_dir
        packaged.append(object_id)
        print(f"  -> {out_root / object_id}")

    # Preliminary catalog (template annotations) so the gate's descriptors
    # resolve offline; rewritten with the gate survivors below.
    prelim_entries = [
        dict(object_id=object_id, catalog_entry=annotate_object(
            upstream_dirs[object_id], 0, 0)[2])
        for object_id in packaged
    ]
    write_catalog(out_root, prelim_entries)

    if gate:
        print(f"\nGenesis sanity gate on {len(packaged)} object(s) ...")
        # The gate worker is a separate process, so the oversized-asset limit
        # travels through the environment (see run_sanity_gate).
        os.environ["ARTVIP_MAX_SIZE_M"] = str(max_size_m)
        kept, passed_records, dropped_records = _run_sanity_gate_batched(
            __file__, run_sanity_gate, output_dir, packaged,
            render_dir=render_dir,
            bbox_tol_abs=bbox_tol_abs,
            bbox_tol_rel=bbox_tol_rel,
        )
        gate_entries: List[dict] = []
        for record in passed_records:
            record["n_files"] = len(list((out_root / record["object_id"]).rglob("*")))
            entry = annotate_object(
                upstream_dirs[record["object_id"]],
                sum(1 for j in record["joints"] if j["type"] == "REVOLUTE"),
                sum(1 for j in record["joints"] if j["type"] == "PRISMATIC"),
            )[2]
            entry["tags"] = sorted(set(entry["tags"]) | {"articulated"})
            gate_entries.append(
                dict(object_id=record["object_id"], catalog_entry=entry)
            )
        write_catalog(
            out_root, gate_entries,
            drop=[record["object_id"] for record in dropped_records],
        )
    else:
        kept, passed_records, dropped_records = list(packaged), [], []

    (out_root / "LICENSE").write_text(LICENSE_TEXT)
    (out_root / "README.md").write_text(PROVENANCE_README)
    (out_root / "packaging_log.json").write_text(json.dumps(dict(
        source=str(source),
        categories=list(categories),
        packaged=packaged,
        passed=passed_records,
        dropped=dropped_records,
    ), indent=1, ensure_ascii=False) + "\n")

    print(f"\nPrepared {len(kept)} object(s) in {out_root}")
    if dropped_records:
        print("Dropped object(s) failing the sanity gate / repackage:")
        for record in dropped_records:
            print(f"  - {record['object_id']}: {record.get('error')}")
    return out_root, kept


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _fetch_dataset(categories: Sequence[str]) -> Path:
    """Snapshot-download the upstream dataset subsets (HF-cached)."""
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            HF_DATASET_REPO,
            repo_type="dataset",
            allow_patterns=[
                *(f"{UPSTREAM_OBJECTS_DIR}/{category}/**" for category in categories),
                "README.md",
            ],
        )
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source", type=Path, default=None,
        help="Existing ArtVIP dataset snapshot (with Articulated_objects/). "
             f"If omitted, the dataset is fetched from {HF_DATASET_REPO} into "
             "the huggingface_hub cache (only the selected categories).",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to place the prepared 'objects/artvip' tree in.",
    )
    parser.add_argument(
        "--categories", default=",".join(DEFAULT_CATEGORIES),
        help="Comma-separated subset of ArtVIP top-level categories "
             f"(default: {', '.join(DEFAULT_CATEGORIES)}; known: "
             f"{', '.join(ALL_KNOWN_CATEGORIES)}). EXCLUDED_CATEGORIES are "
             "refused (defaults and opt-in alike) pending license confirmation.",
    )
    parser.add_argument(
        "--objects", type=str, default=None,
        help="Comma-separated subset of object ids (packaged or upstream "
             "spellings, substring match) to ingest (default: all in the "
             "selected categories).",
    )
    parser.add_argument(
        "--skip-gate", action="store_true",
        help="Skip the Genesis GPU sanity-load gate (loader + joints + bbox + render).",
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="Skip the per-object thumbnail render (the gate still checks loads + bboxes).",
    )
    parser.add_argument(
        "--render-dir", type=Path, default=Path("/tmp/artvip_sanity"),
        help="Directory for the per-object eyeball renders (default /tmp/artvip_sanity).",
    )
    parser.add_argument(
        "--bbox-tol-abs", type=float, default=BBOX_TOL_ABS,
        help=f"Absolute bbox cross-check tolerance in metres (default {BBOX_TOL_ABS}).",
    )
    parser.add_argument(
        "--bbox-tol-rel", type=float, default=BBOX_TOL_REL,
        help=f"Relative bbox cross-check tolerance, as a fraction of the "
             f"per-axis extent (default {BBOX_TOL_REL}).",
    )
    parser.add_argument(
        "--max-size-m", type=float, default=MAX_STAGE_DIAGONAL_M,
        help="Oversized-asset guard: drop (before the expensive Genesis "
             "load) any object whose stage world AABB diagonal exceeds this "
             f"many metres (default {MAX_STAGE_DIAGONAL_M:.1f}).",
    )
    args = parser.parse_args(argv)

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    unknown = [c for c in categories if c not in ALL_KNOWN_CATEGORIES]
    if unknown:
        raise SystemExit(
            f"Unknown ArtVIP category/ies: {unknown}. Known: "
            f"{list(ALL_KNOWN_CATEGORIES)}."
        )
    # No categories are excluded: all are packaged under the upstream license.

    if args.source is not None:
        source = args.source.expanduser().resolve()
        if not (source / UPSTREAM_OBJECTS_DIR).is_dir():
            raise SystemExit(
                f"No '{UPSTREAM_OBJECTS_DIR}/' directory in --source: {source}"
            )
    else:
        print(f"Fetching {HF_DATASET_REPO} ({', '.join(categories)}) via huggingface_hub ...")
        source = _fetch_dataset(categories)
        print(f"  -> {source}")

    out_root, _kept = prepare(
        source,
        args.output_dir.expanduser().resolve(),
        categories,
        objects=(
            [o for o in args.objects.split(",") if o.strip()] if args.objects else None
        ),
        gate=not args.skip_gate,
        bbox_tol_abs=args.bbox_tol_abs,
        bbox_tol_rel=args.bbox_tol_rel,
        render_dir=None if args.no_render else args.render_dir,
        max_size_m=args.max_size_m,
    )

    print(
        "\nTo upload (requires write access to the org):\n"
        f"  hf upload {HF_UPLOAD_REPO} \\\n"
        f"      {out_root} {OUTPUT_ASSETS_SUBDIR} --repo-type dataset\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
