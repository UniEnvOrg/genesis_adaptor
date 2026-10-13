#!/usr/bin/env python3
# Upstream asset license: CC BY 4.0 (attribution only) per the
# nvidia/PhysicalAI-SimReady-Warehouse-01 dataset card; the archive commits no
# LICENSE file.  This packager only rehosts the USD payloads unchanged; the
# original license applies to the packaged assets.  Attribution required.
"""Prepare NVIDIA SimReady Warehouse rigid props for the collection's assets repo.

Downloads (or reuses) the `nvidia/PhysicalAI-SimReady-Warehouse-01
<https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01>`_
dataset and repackages its per-object props into the layout expected by
``unienv_genesis_collection.objects.hub_objects.articulated.simready_warehouse``
(one self-contained directory per prop, loaded with ``gs.morphs.USD`` via
``scene.add_stage``)::

    <output>/objects/simready_warehouse/LICENSE                  # CC BY 4.0 attribution
    <output>/objects/simready_warehouse/README.md                # provenance note
    <output>/objects/simready_warehouse/catalog.json             # CSV-derived annotations + tags
    <output>/objects/simready_warehouse/packaging_log.json       # per-prop gate log
    <output>/objects/simready_warehouse/<object_id>/<object_id>.usd
    <output>/objects/simready_warehouse/<object_id>/materials/...# textures sidecar

**Selection uses the dataset's 753-row CSV catalog**
(``physical_ai_simready_warehouse_01.csv``; asset_name / relative_path /
classification / label / mass).  ``--categories`` filters by the CSV
classification class — ``prop`` (475 "Prop general" rows, excluding the
hand-manipulation subset), ``handmanip`` (175 rows) and ``assembly``
(103 rows, opt-in).  Object ids derive from the CSV asset names.

**RigidBodyAPI authoring (mandatory).**  Upstream ships colliders only ("no
Rigid Bodies are applied" — the README's Isaac recipe is *select the asset
and add Rigid Body*).  Packaging authors ``UsdPhysics.RigidBodyAPI`` on the
object's root payload prim (``/RootNode``), which makes ``scene.add_stage``
collapse the prop into exactly **one** rigid entity (verified: without it,
per-prim ``CollisionAPI`` colliders parse as one entity each, and
``ArticulationRootAPI`` alone yields no entity at all).  The packaged stage
is the upstream ``<Asset>_physics.usd`` overlay variant (renamed to
``<object_id>.usd``; it sublayers the visual layer and carries the
convex-hull colliders over the same prim paths).

**MDL → UsdPreviewSurface** (shared converter): upstream shading is OmniPBR
MDL (local ``materials/*.mdl`` recipes or the inline OmniPBR graph); the
rewrite re-wires diffuse/ORM/normal/roughness into UsdPreviewSurface and
drops the ``materials/*.mdl`` files (their textures ship).

**Units.**  Upstream ``metersPerUnit`` is mixed (0.01 / 0.0099999… / 1.0);
each packaged stage is normalised to 1.0 by folding the old factor into the
root prim's linear map (a root-level uniform scale) — identical world size,
one unit convention.

**License — CC BY 4.0, attribution only.**  The dataset card tags
``license:cc-by-4.0`` but the archive commits no LICENSE file; the
attribution README ships next to the assets noting that.

Usage::

    python scripts/prepare_simready_warehouse_assets.py \\
        --output-dir /tmp/wh_assets --categories prop --objects Cardbox_C1
    # --source stays optional: without it the CSV is fetched, the selected
    # props resolved, and each prop's subtree is downloaded per object
    # (never the 14.45 GB whole dataset).
    # or with an already-downloaded snapshot (the Props/ tree):
    python scripts/prepare_simready_warehouse_assets.py \\
        --source <snapshot> --output-dir /tmp/wh_assets

Then upload to the Hub (needs write access to the org)::

    hf upload UniEnvOrg/unienv_genesis_collection_assets \\
        /tmp/wh_assets/objects/simready_warehouse objects/simready_warehouse --repo-type dataset
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


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
    sanity_check_rigid,
)
from _packaging.usd_verify import verify_object_assets as _verify_object_assets  # noqa: E402


HF_DATASET_REPO = "nvidia/PhysicalAI-SimReady-Warehouse-01"
CSV_NAME = "physical_ai_simready_warehouse_01.csv"
UPSTREAM_PROPS_DIR = "Props"
# Shared physics materials (physics_metal.usda / physics_glass.usda / …) that
# per-prop stages sublayer; lives outside the per-prop subtrees.
UPSTREAM_PHYSICS_MATERIALS_DIR = "Props/materials/physics"
OUTPUT_ASSETS_SUBDIR = "objects/simready_warehouse"
HF_UPLOAD_REPO = "UniEnvOrg/unienv_genesis_collection_assets"

# CSV classification → packaged category tag.  The upstream rows read
# "Prop general", "Prop general hand manipulation" and "Assembly".
CATEGORY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "prop": ("Prop general",),
    "handmanip": ("Prop general hand manipulation",),
    "hand-manip": ("Prop general hand manipulation",),
    "assembly": ("Assembly",),
}
DEFAULT_CATEGORIES: Tuple[str, ...] = ("prop", "handmanip")
ALL_KNOWN_CATEGORIES: Tuple[str, ...] = tuple(sorted(CATEGORY_ALIASES))

# Bbox cross-check tolerances (offline-FK box proxy vs real mesh geometry)
# — defined once in ``_packaging.sanity`` (BBOX_TOL_ABS/BBOX_TOL_REL) and
# imported above.

LICENSE_TEXT = """\
NVIDIA Physical AI SimReady Warehouse OpenUSD assets.

Dataset: "NVIDIA Physical AI SimReady Warehouse OpenUSD Dataset" (v1.1.0)
— nvidia/PhysicalAI-SimReady-Warehouse-01,
https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01

License: **CC-BY-4.0** (Creative Commons Attribution 4.0 International) per
the dataset card (https://creativecommons.org/licenses/by/4.0/).  NOTE: the
dataset archive commits **no LICENSE file**; the dataset card is the
authoritative license statement — consumers should verify it and attribute
"Physical AI SimReady Warehouse Dataset, © NVIDIA Corporation, CC BY 4.0"
when using these models.

Repackaged by genesis_adaptor/scripts/prepare_simready_warehouse_assets.py.
"""

PROVENANCE_README = """\
# NVIDIA SimReady Warehouse rigid props

Repackaged from the [Physical AI SimReady Warehouse dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01)
(NVIDIA's industrial-warehouse OpenUSD props, CC BY 4.0): one self-contained
directory per prop under `objects/simready_warehouse/<object_id>/`,
containing the packaged stage (`<object_id>.usd` — the upstream
`<Asset>_physics.usd` overlay renamed, which sublayers the visual layer and
carries the convex-hull colliders) plus the `materials/` texture sidecar.

> **License note.**  The dataset card tags **CC BY 4.0** but the archive
> commits **no LICENSE file** — see `LICENSE` and consumers should verify
> the dataset card before assuming the grant.  Attribute "Physical AI
> SimReady Warehouse Dataset, © NVIDIA Corporation (CC BY 4.0)".

* **Rigid props, zero articulation upstream** — the dataset README states
  "USDPhysics Colliders applied, but no Rigid Bodies"; packaging authors
  `UsdPhysics.RigidBodyAPI` on the object's root payload prim (verified:
  `scene.add_stage` collapses the prop into exactly one entity; per-prim
  CollisionAPI colliders would split into one entity each, and
  ArticulationRootAPI alone yields none).  Jointless is the point: the
  descriptor exposes an empty moving-joint set and a welded root by
  default (`fixed=False` leaves the prop dynamic).
* **Materials are rewritten from OmniPBR MDL into native UsdPreviewSurface**
  (`unienv_genesis_collection.utils.usd_conversions.convert_mdl_to_preview_surface`
  — usd-core cannot read `.mdl`); the local `materials/*.mdl` recipes are
  dropped, their textures ship.
* **Units normalised to metersPerUnit = 1.0** (upstream mixes 0.01 /
  0.0099999… / 1.0) by folding the factor into the root prim's linear map.
* Annotations/tags derive from the dataset CSV catalog (`classification`,
  `label`, `mass`).
* Every shipped prop passed the Genesis sanity gate: single-entity
  `add_stage` load (no moving joints required — rigid props), and the
  offline box-proxy FK union bbox cross-checked against
  `entity.get_vAABB()`.  Per-prop details are recorded in
  `packaging_log.json`.

When using these models, please attribute the dataset — see `LICENSE` and
the license note above.
"""


# ---------------------------------------------------------------------------
# CSV selection
# ---------------------------------------------------------------------------

def read_csv_catalog(csv_file: Path) -> List[dict]:
    """The dataset's CSV rows (only the per-prop rows under ``Props/``)."""
    rows = []
    with open(csv_file, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            relative = (row.get("relative_path") or "").strip()
            if not relative or not relative.startswith(f"{UPSTREAM_PROPS_DIR}/"):
                continue  # scene / SubLayers rows are not per-object props
            rows.append(row)
    return rows


def _csv_object_id(relative_path: str) -> str:
    """Packaged id from a CSV row's relative_path.

    "Prop general" rows point at the prop's ``<Asset>_physics.usd`` overlay —
    the object is its parent directory (``Props/general/Cardbox_C1/…`` →
    ``prop_cardbox_c1``).  HandManipulation rows point at one specific stage
    file (several variants share a directory), so the file stem names the
    object (``Props/general/HandManipulation/book_open_a/sm_book_open_a01_…``
    → ``hand_sm_book_open_a01_simready_01``).
    """
    parent = relative_path.rsplit("/", 1)[0]
    if "HandManipulation" in parent:
        stem = Path(relative_path).name.removesuffix(".usd")
        prefix = "hand"
    else:
        stem = Path(parent).name
        prefix = "prop"
    slug = "".join(char if char.isalnum() else "_" for char in stem).lower()
    return f"{prefix}_{slug}"


def select_csv_objects(rows: Sequence[dict], categories: Sequence[str]) -> List[Tuple[str, dict]]:
    """Group CSV rows into ``(object_id, record)`` per prop dir.

    The record keeps the first row's classification/label/q_code/mass.
    Classifications outside *categories* are skipped.
    """
    accepted = set()
    for category in set(categories):
        accepted.update(CATEGORY_ALIASES.get(category, (category,)))
    found: Dict[str, dict] = {}
    for row in rows:
        classification = (row.get("classification") or "").strip()
        if classification not in accepted:
            continue
        object_id = _csv_object_id(row["relative_path"])
        if object_id in found:
            continue  # the visual + `_physics` rows name the same prop dir
        found[object_id] = dict(
            classification=classification,
            label=(row.get("label") or "").strip(),
            q_code=(row.get("q_code") or "").strip(),
            mass=(row.get("mass") or "").strip(),
            asset_name=(row.get("asset_name") or "").strip(),
        )
    return sorted((object_id, found[object_id]) for object_id in found)


def select_records(rows: Sequence[dict], categories: Sequence[str],
                   objects: Optional[Sequence[str]] = None) -> List[Tuple[str, dict]]:
    """CSV rows → the ``(object_id, record)`` selection of one run.

    The single selection path, shared by the packaging and by the
    no-``--source`` download planning (which needs the same answer *before*
    the files exist), so both always agree on which props a run covers.
    """
    records = select_csv_objects(rows, categories)
    if objects:
        wanted = {o.strip().lower() for o in objects if o.strip()}
        selected = [
            (object_id, record)
            for object_id, record in records
            if (object_id.lower() in wanted
                or any(w in object_id.lower()
                       or w in (record.get("asset_name") or "").lower()
                       for w in wanted))
        ]
        if not selected:
            raise SystemExit(
                f"--objects matched none of the {len(records)} CSV props."
            )
        return selected
    if not records:
        raise SystemExit(f"No props in the CSV catalog for {list(categories)}.")
    return list(records)


def _fetch_catalog(output_dir: Path) -> Path:
    """The dataset's CSV catalog, cached under *output_dir*."""
    from huggingface_hub import hf_hub_download

    local_csv = Path(hf_hub_download(HF_DATASET_REPO, CSV_NAME, repo_type="dataset"))
    csv_file = output_dir / CSV_NAME
    if not csv_file.is_file():
        shutil.copy2(local_csv, csv_file)
    return csv_file


# ---------------------------------------------------------------------------
# Repackaging (physics overlay + RigidBodyAPI + units + materials)
# ---------------------------------------------------------------------------

def _top_level_prims(stage) -> List:
    """The pseudo-root's direct children (the object root payload prims)."""
    return list(stage.GetPseudoRoot().GetAllChildren())


def _normalise_units(stage) -> float:
    """Normalise the stage to ``metersPerUnit = 1.0`` (returns the old factor).

    USD positions are authored in stage units; folding the factor into each
    top-level prim's linear map scales the whole composed tree uniformly —
    the world size (in metres) is unchanged, only the unit convention
    changes (so Genesis's ``_meter_scale`` correction becomes a no-op and
    the offline FK matches the runtime frame exactly).
    """
    from pxr import Gf, Usd, UsdGeom

    old = float(UsdGeom.GetStageMetersPerUnit(stage))
    if abs(old - 1.0) <= 1e-6:
        return old
    for root in _top_level_prims(stage):
        if not root.IsA(UsdGeom.Xformable):
            continue
        xform = UsdGeom.Xformable(root)
        scale_ops = [
            op for op in xform.GetOrderedXformOps()
            if op.GetOpType() == UsdGeom.XformOp.TypeScale
        ]
        if scale_ops:
            scale_op = scale_ops[0]
            value = scale_op.Get() or Gf.Vec3d(1.0, 1.0, 1.0)
            scale_op.Set(
                Gf.Vec3d(*(float(channel) * old for channel in value)),
                Usd.TimeCode.Default(),
            )
        else:
            # A uniform scale appended last is outermost — it commutes with
            # any authored rotation and covers the whole subtree.
            appended = xform.AddScaleOp()
            appended.Set(Gf.Vec3f(old, old, old), Usd.TimeCode.Default())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    return old


def _apply_root_rigid_body(stage) -> int:
    """Author ``UsdPhysics.RigidBodyAPI`` on the object root payload prim(s).

    The dataset ships colliders only; the README's Isaac recipe is applied
    offline on the top-level payload Xform (NOT the geometry or material
    prims within it).  For assemblies the payload'd props compose under the
    same root, so the single API collapses the whole arrangement into one
    welded rigid body.  Returns the number of authored APIs.
    """
    from pxr import UsdPhysics

    count = 0
    for prim in _top_level_prims(stage):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        UsdPhysics.RigidBodyAPI.Apply(prim)
        count += 1
    return count


def _copy_assembly_closure(source_root: Path, source_rel_dir: str, physics_file_name: str,
                           layout_root: Path) -> List[Path]:
    """Copy an assembly's payload/reference closure into the packaged layout.

    Assembly stages payload their constituent props by *relative* paths that
    reach outside the assembly directory (``../../general/Cardbox_A1/…``),
    so the packaged tree preserves the upstream subtree layout under the
    object directory (``<layout_root>/Props/assembly/<Asset>/…`` with the
    payload targets at ``<layout_root>/Props/general/<Prop>/…``).  The
    closure is walked with ``UsdUtils.ExtractExternalReferences`` (payload
    arcs surface as references) and every referenced file is copied **with
    its whole directory** (the props' `materials/` texture sidecars ride
    along; the `.thumbs` previews and `*.mdl` recipes are pruned afterwards,
    exactly like the non-assembly path).

    Returns the copied closure stage files (outside the assembly's own dir).
    """
    from pxr import UsdUtils

    source_root = source_root.resolve()
    copied: List[Path] = []
    pending = [source_root / source_rel_dir / physics_file_name]
    seen = {pending[0].resolve()}
    while pending:
        layer_file = pending.pop()
        try:
            sublayers, assets, references = UsdUtils.ExtractExternalReferences(str(layer_file))
        except Exception:  # noqa: BLE001 — the loader reports unreadable layers
            continue
        for reference in list(references) + list(sublayers):
            resolved = (layer_file.parent / reference).resolve()
            if not resolved.is_file():
                continue
            try:
                resolved.relative_to(source_root)
            except ValueError:
                continue  # outside the snapshot (should not happen)
            if resolved.resolve() in seen:
                continue
            seen.add(resolved.resolve())
            dest_dir = layout_root / resolved.parent.relative_to(source_root)
            shutil.copytree(resolved.parent, dest_dir, dirs_exist_ok=True)
            # The assembly path copies whole directories (the props'
            # `materials/` texture sidecars ride along); prune the same
            # payload junk the non-assembly path drops (`.thumbs` previews
            # and the `*.mdl` recipes whose textures ship alone).
            for junk in list(dest_dir.rglob(".thumbs")) + list(dest_dir.rglob("*.mdl")):
                if junk.is_dir():
                    shutil.rmtree(junk, ignore_errors=True)
                else:
                    junk.unlink(missing_ok=True)
            copied.append(dest_dir / resolved.name)
            pending.append(dest_dir / resolved.name)
    return copied


def _thin_sublayer_layer(description_file: Path, sublayer_rel: str, default_prim: str) -> None:
    """Author the assembly's thin packaged root layer.

    A sublayer-only root layer composes the preserved assembly stage; the
    root layer's own metadata (``metersPerUnit``) wins over the sublayer's,
    which is what makes the unit normalisation hold for the composed stage.
    """
    description_file.write_text(
        "#usda 1.0\n"
        "(\n"
        f'    defaultPrim = "{default_prim}"\n'
        "    metersPerUnit = 1\n"
        '    upAxis = "Z"\n'
        f"    subLayers = [@./{sublayer_rel}@]\n"
        ")\n"
    )


def repackage_object(source_root: Path, relative_paths: Sequence[str], object_id: str,
                     out_root: Path, record: Optional[dict] = None) -> Tuple[Path, int, float]:
    """Repackage one prop into ``out_root/<object_id>/``.

    Copies the object's directory verbatim (visual stage + `_physics`
    overlay + `materials/` sidecars, minus `.thumbs` and the upstream
    `*.mdl` recipes).  The packaged description file is the physics overlay
    renamed to ``<object_id>.usd`` when one exists (it sublayers the visual
    layer and carries the convex-hull colliders over the same prim paths);
    hand-manipulation props ship their own visual stage instead (they carry
    no `_physics` overlay upstream).  Then applies:

    1. the **MDL → UsdPreviewSurface** rewrite (materials are unreadable to
       usd-core upstream), which also rewrites ``<UDIM>`` texture templates
       to their concrete first tile (usd-core does not expand ``<UDIM>``);
    2. ``UsdPhysics.RigidBodyAPI`` on the object root (the README's
       "add rigid body" recipe — collapses the prop to one entity);
    3. the ``metersPerUnit`` normalisation to 1.0;
    4. the **degenerate-mesh hide** (sub-4-point meshes cannot form a convex
       hull and the loader's name-pattern fallback makes them collision
       geoms — see :func:`hide_degenerate_mesh_prims`).

    **Assemblies** (``Props/assembly/…``) compose their constituent props by
    payload, so their tree is copied *preserving the upstream subtree layout*
    plus every payload-closure prop directory (both pruned of `.thumbs` and
    the `*.mdl` recipes exactly like the non-assembly path); the packaged
    root layer is a thin sublayer of the preserved assembly stage.  The
    single-body collapse applies to the composed assembly (one welded
    arrangement).

    Verifies that every file the pxr layer stack + texture references name
    exists in the packaged tree.  Returns ``(packaged_dir, n_rigid, mpu)``.
    """
    from pxr import Usd

    from unienv_genesis_collection.utils.usd_conversions import (
        convert_mdl_to_preview_surface,
        hide_degenerate_mesh_prims,
        strip_unresolvable_physics_references,
    )

    relative = relative_paths[0]
    source_rel_dir = relative.rsplit("/", 1)[0]
    asset_dir = source_root / source_rel_dir
    if not asset_dir.is_dir():
        raise FileNotFoundError(f"No prop directory in the source snapshot: {asset_dir}")
    physics_files = sorted(p for p in asset_dir.glob("*_physics.usd") if p.is_file())
    physics_file = physics_files[0] if physics_files else None
    is_assembly = source_rel_dir.startswith(f"{UPSTREAM_PROPS_DIR}/assembly/")

    obj_out_dir = out_root / object_id
    obj_out_dir.mkdir(parents=True, exist_ok=True)
    if is_assembly:
        # Preserve the upstream subtree layout so the payload arcs resolve.
        layout_dir = obj_out_dir / source_rel_dir
        shutil.copytree(asset_dir, layout_dir, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(".thumbs", "*.mdl"))
        closure_files = _copy_assembly_closure(
            source_root, source_rel_dir, physics_file.name, obj_out_dir,
        )
        for stage_file in [layout_dir / physics_file.name, *closure_files]:
            convert_mdl_to_preview_surface(str(stage_file))
            # The assembly stage composes its payload'd props as (possibly
            # instanceable) prims that are read-only to edit on the composed
            # stage — the degenerate-mesh rewrite is applied to each closure
            # prop stage file instead, where the mesh prim is plainly
            # editable and the edit composes back into the assembly.
            hide_degenerate_mesh_prims(str(stage_file))
        preserved = layout_dir / physics_file.name
        stage = Usd.Stage.Open(str(preserved))
        n_rigid = _apply_root_rigid_body(stage)
        mpu = _normalise_units(stage)
        stage.GetRootLayer().Save()
        reopened = Usd.Stage.Open(str(preserved))
        default_prim = reopened.GetDefaultPrim()
        default_prim_name = str(default_prim.GetName()) if default_prim.IsValid() else _top_level_prims(reopened)[0].GetName()
        packaged_stage = obj_out_dir / f"{object_id}.usd"
        _thin_sublayer_layer(
            packaged_stage, f"{source_rel_dir}/{physics_file.name}", default_prim_name,
        )
        _verify_object_assets(packaged_stage, obj_out_dir)
        return obj_out_dir, n_rigid, mpu

    for path in sorted(asset_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(asset_dir)
        if ".thumbs" in rel.parts:
            continue
        if rel.suffix.lower() == ".mdl":
            continue  # the MDL recipe is rewritten, its textures ship
        if physics_file is not None and path.name == physics_file.name:
            continue  # the overlay is packaged (renamed) as the description file
        (obj_out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, obj_out_dir / rel)

    packaged_stage = obj_out_dir / f"{object_id}.usd"
    if physics_files:
        # The packaged root is the physics overlay (renamed); its `subLayers`
        # still points at the visual file, which keeps its upstream name.
        shutil.copy2(physics_files[0], packaged_stage)
        visual_layer = packaged_stage.parent / (
            physics_files[0].name.removesuffix("_physics.usd") + ".usd"
        )
        if visual_layer.is_file():
            convert_mdl_to_preview_surface(str(visual_layer))
    else:
        # Hand-manipulation props: the row names the stage file itself.
        present = [obj_out_dir / Path(p).name for p in relative_paths
                   if (obj_out_dir / Path(p).name).is_file()]
        if not present:
            raise FileNotFoundError(
                f"No stage file for {object_id} (rows: {relative_paths})."
            )
        present[0].rename(packaged_stage)

    convert_mdl_to_preview_surface(str(packaged_stage))
    # Strip the upstream physics-material prims whose internal reference is an
    # absolute ``omniverse://`` URL (Omniverse Nucleus only — never resolvable
    # off the filesystem, so the verifier would flag them as missing).  The
    # pipeline re-authors its own RigidBodyAPI + convex colliders below and
    # Genesis re-derives the physics material, so the reference is dead weight.
    strip_unresolvable_physics_references(str(packaged_stage))
    # Hide the degenerate (< 4-point) mesh prims the dataset ships on a few
    # props: Genesis's loader's name-pattern fallback makes such a prim BOTH
    # visual and collision regardless of ``UsdPhysics.CollisionAPI``, and the
    # qhull convex hull of 3 points raises QH6214 — killing the whole load.
    hide_degenerate_mesh_prims(str(packaged_stage))
    stage = Usd.Stage.Open(str(packaged_stage))
    n_rigid = _apply_root_rigid_body(stage)
    mpu = _normalise_units(stage)
    stage.GetRootLayer().Save()

    _verify_object_assets(packaged_stage, obj_out_dir)
    return obj_out_dir, n_rigid, mpu



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
    keep the entries of objects that are not part of this run; a re-run of
    the same object rewrites its entry.  The log line prints the family's
    "rigid props" summary (the packager's observable catalog label).
    """
    return _write_catalog_impl(
        out_root, records, merge=merge, drop=drop, overrides=CATALOG_OVERRIDES,
        summary_label="rigid props",
    )

CATALOG_OVERRIDES: Dict[str, dict] = {}


def annotate_object(record: dict) -> Tuple[str, dict]:
    """One catalog entry from the CSV record (classification/label/mass).

    Annotations broad → detailed: ``[kind, "a <label lowercased>", "… (N kg)"]``
    — the CSV label is the WikiData-anchored English class name; tags carry
    the packaged category (``prop``/``hand``/``assembly``), the label token
    and ``rigid``.
    """
    classification = str(record.get("classification") or "")
    category = "hand" if "hand manipulation" in classification.lower() else (
        "assembly" if classification.lower() == "assembly" else "prop"
    )
    label = str(record.get("label") or "warehouse prop").strip()
    words = label.replace("_", " ").lower()
    annotations = [f"warehouse {category}", f"a {words}"]
    mass = str(record.get("mass") or "").strip()
    if mass:
        annotations.append(f"a {words} of about {mass} kilograms")
    entry_tags = {category, "rigid", "warehouse"}
    entry_tags.update(
        token for token in words.replace("-", "_").replace("  ", " ").split() if token
    )
    entry = {
        "language_annotations": list(dict.fromkeys(annotations)),
        "tags": sorted(entry_tags),
    }
    return category, entry


# ---------------------------------------------------------------------------
# Genesis sanity gate (GPU) — the ArtVIP gate with the joint set optional
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Genesis sanity gate (GPU) — the shared loop (``_packaging.sanity``) wired to
# this family's descriptor factory + the rigid-prop check
# ---------------------------------------------------------------------------

def _gate_descriptor(object_id: str):
    """Build the family descriptor for one packaged id (the only per-family
    part of the gate).  Imported lazily so the asset repointing
    (``asset_path.ASSETS_DIR`` → *output_dir*) is already in place."""
    from unienv_genesis_collection.objects.hub_objects.articulated.simready_warehouse import (
        SimReadyWarehouseProp,
    )

    return SimReadyWarehouseProp(object_id)


def run_sanity_gate(
    output_dir: Path,
    object_ids: Sequence[str],
    render_dir: Optional[Path] = None,
    bbox_tol_abs: float = BBOX_TOL_ABS,
    bbox_tol_rel: float = BBOX_TOL_REL,
) -> Tuple[List[str], List[dict], List[dict]]:
    """Sanity-load every packaged prop; drop (warn on) the failing ones."""
    return _run_sanity_gate_impl(
        output_dir, object_ids,
        descriptor_factory=_gate_descriptor,
        check=sanity_check_rigid,
        render_dir=render_dir,
        bbox_tol_abs=bbox_tol_abs,
        bbox_tol_rel=bbox_tol_rel,
        asset_noun="prop",
    )

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def prepare(
    source: Path,
    output_dir: Path,
    categories: Sequence[str],
    csv_file: Path,
    objects: Optional[Sequence[str]] = None,
    gate: bool = True,
    bbox_tol_abs: float = BBOX_TOL_ABS,
    bbox_tol_rel: float = BBOX_TOL_REL,
    render_dir: Optional[Path] = None,
) -> Tuple[Path, List[str]]:
    """Package the selected CSV classes into ``output_dir/objects/simready_warehouse``."""
    out_root = output_dir / OUTPUT_ASSETS_SUBDIR
    out_root.mkdir(parents=True, exist_ok=True)

    rows = read_csv_catalog(csv_file)
    selected = select_records(rows, categories, objects)

    object_records: Dict[str, dict] = {}
    object_rows: Dict[str, List[str]] = {}
    packaged: List[str] = []
    print(f"Repackaging {len(selected)} prop(s) ...")
    for object_id, record in selected:
        # All CSV rows naming this object (the visual + physics pair).
        group_rows = [
            row["relative_path"] for row in rows
            if _csv_object_id(row["relative_path"]) == object_id
        ]
        try:
            repackage_object(source, group_rows, object_id, out_root, record)
        except Exception as exc:  # noqa: BLE001 — repackage failure ⇒ drop
            print(f"  [repackage] {object_id}: DROPPED — {type(exc).__name__}: {exc}")
            shutil.rmtree(out_root / object_id, ignore_errors=True)
            continue
        object_records[object_id] = record
        object_rows[object_id] = group_rows
        packaged.append(object_id)
        print(f"  -> {out_root / object_id}")

    prelim_entries = [
        dict(object_id=object_id, catalog_entry=annotate_object(
            object_records[object_id])[1])
        for object_id in packaged
    ]
    write_catalog(out_root, prelim_entries)

    if gate:
        print(f"\nGenesis sanity gate on {len(packaged)} prop(s) ...")
        kept, passed_records, dropped_records = _run_sanity_gate_batched(
            __file__, run_sanity_gate, output_dir, packaged,
            render_dir=render_dir,
            bbox_tol_abs=bbox_tol_abs,
            bbox_tol_rel=bbox_tol_rel,
        )
        gate_entries: List[dict] = []
        for record in passed_records:
            record["n_files"] = len(list((out_root / record["object_id"]).rglob("*")))
            entry = annotate_object(object_records[record["object_id"]])[1]
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

    print(f"\nPrepared {len(kept)} prop(s) in {out_root}")
    if dropped_records:
        print("Dropped prop(s) failing the sanity gate / repackage:")
        for record in dropped_records:
            print(f"  - {record['object_id']}: {record.get('error')}")
    return out_root, kept


# ---------------------------------------------------------------------------
# Download (HF dataset)
# ---------------------------------------------------------------------------

def fetch_dataset(output_dir: Path,
                  relative_paths: Optional[Sequence[str]] = None) -> Path:
    """Fetch the selected props' subtrees from the Hub into *output_dir*.

    Without ``--source`` the packaging still needs the actual USD files, so
    the dataset's relative layout is materialised under ``<output_dir>/…``
    with per-subtree downloads (each prop directory is a few MB; the 14.45 GB
    dataset is never pulled whole).  Every directory the CSV rows name gets
    one ``snapshot_download`` pass with a ``<dir>/**`` ``allow_patterns``, so
    the ``materials/`` texture sidecars and the visual layers the rows do not
    name ride along at their upstream relative paths.

    Assemblies compose their constituent props by relative payload arcs, so
    the plan is extended with every payload directory the downloaded
    assembly stages reference (:func:`_payload_closure_patterns`) until the
    closure closes — an assembly packs complete, never as a dangling shell.
    Idempotent: hf snapshot semantics skip already-local files.

    Args:
        output_dir: Root the upstream relative layout is materialised into.
        relative_paths: The CSV rows' ``relative_path`` values naming the
            files to fetch (visual + ``_physics`` stages).  ``None``/empty
            fetches nothing (selection-only runs).

    Returns:
        *output_dir*, the shadow snapshot to repackage from.
    """
    plan = _hub_fetch_plan(relative_paths)
    # Shared physics layers (``physics_metal.usda`` etc.) live in the
    # ``Props/materials/physics/`` subtree, outside every per-prop directory,
    # so the ``<dir>/**`` patterns never pull them — but many props sublayer
    # them.  Add the subtree once up front.
    plan.setdefault(UPSTREAM_PHYSICS_MATERIALS_DIR, (f"{UPSTREAM_PHYSICS_MATERIALS_DIR}/**",))
    fetched: Dict[str, Tuple[str, ...]] = {}
    for _round in range(4):
        pending = {d: p for d, p in plan.items() if d not in fetched}
        if pending:
            _download_subtrees(output_dir, pending)
            fetched.update(pending)
        extra = _payload_closure_patterns(output_dir, plan)
        if not extra:
            break
        plan.update(extra)
    else:
        print(
            "[warn] the assembly payload closure is still incomplete after "
            "4 download rounds — the affected assemblies will be dropped by "
            "the repackage step (see its per-prop message)."
        )
    return output_dir


def _download_subtrees(output_dir: Path, plan: Dict[str, Tuple[str, ...]]) -> None:
    """Fetch every planned prop subtree in a single ``snapshot_download`` pass.

    One pass with a combined ``allow_patterns`` keeps the Hub API call count
    at ~1 per download round (a per-directory loop would fire hundreds of
    ``tree`` listings and trip the 1000-requests/5-min rate limit, as it did
    on the first full run).  ``snapshot_download`` matches patterns
    client-side, so the per-prop ``<dir>/**`` patterns need no widening.
    """
    all_patterns = sorted({p for patterns in plan.values() for p in patterns})
    print(
        f"Fetching {HF_DATASET_REPO}: {len(plan)} prop subtree(s) "
        f"({len(all_patterns)} pattern(s)) in one snapshot pass ..."
    )
    _snapshot_download_with_retry(
        HF_DATASET_REPO,
        repo_type="dataset",
        allow_patterns=all_patterns,
        local_dir=output_dir,
    )


def _snapshot_download_with_retry(*args, attempts: int = 6, **kwargs):
    """``snapshot_download`` with backoff on Hub rate limits / transient 5xx.

    The rate-limit response carries a ``Retry after N seconds`` hint; we honour
    it (plus a small margin) and otherwise back off exponentially.  Anything
    else propagates immediately.
    """
    import re
    import time

    from huggingface_hub import snapshot_download

    delay = 20.0
    for attempt in range(1, attempts + 1):
        try:
            return snapshot_download(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — re-raised after retries
            message = str(exc)
            rate_limited = "429" in message or "Too Many Requests" in message
            transient = any(code in message for code in ("500", "502", "503", "504"))
            if not (rate_limited or transient) or attempt == attempts:
                raise
            match = re.search(r"Retry after (\d+) seconds", message)
            wait = float(match.group(1)) + 10.0 if match else delay
            print(
                f"  [retry {attempt}/{attempts - 1}] Hub throttled the download "
                f"({'rate limit' if rate_limited else 'transient error'}); "
                f"sleeping {wait:.0f}s ..."
            )
            time.sleep(wait)
            delay = min(delay * 2.0, 600.0)


def _hub_fetch_plan(relative_paths: Optional[Sequence[str]]) -> Dict[str, Tuple[str, ...]]:
    """Group the CSV ``relative_path`` values into per-subtree patterns.

    ``Props/general/Cardbox_C1/Cardbox_C1_physics.usd`` →
    ``{"Props/general/Cardbox_C1": ("Props/general/Cardbox_C1/**",)}`` — the
    subtree is taken whole (the packaged stage references its ``materials/``
    sidecars relatively).  Rows outside ``Props/`` (the scene layer) are
    ignored.
    """
    plan: Dict[str, set] = {}
    for relative in relative_paths or ():
        parent = str(relative).rsplit("/", 1)[0]
        if not parent.startswith(f"{UPSTREAM_PROPS_DIR}/"):
            continue
        patterns = plan.setdefault(parent, set())
        patterns.add(f"{parent}/**")
    return {prop_dir: tuple(sorted(patterns)) for prop_dir, patterns in plan.items()}


def _payload_closure_patterns(
    output_dir: Path,
    plan: Dict[str, Tuple[str, ...]],
) -> Dict[str, Tuple[str, ...]]:
    """New ``Props/…/`` subtrees the *downloaded* assembly stages payload.

    Walks ``UsdUtils.ExtractExternalReferences`` over each downloaded
    assembly stage and returns a fetch pattern for every referenced prop
    directory that is not on disk yet (the ``../../general/<Prop>/…``
    payload arcs); the caller loops until this returns empty.
    """
    from pxr import UsdUtils

    extra: Dict[str, set] = {}
    root = output_dir.resolve()
    for prop_dir in list(plan):
        if not prop_dir.startswith(f"{UPSTREAM_PROPS_DIR}/assembly/"):
            continue
        for stage_file in sorted((output_dir / prop_dir).glob("*.usd")):
            try:
                _sublayers, _assets, references = UsdUtils.ExtractExternalReferences(
                    str(stage_file)
                )
            except Exception:  # noqa: BLE001 — unreadable layers are dropped later
                continue
            for reference in references:
                resolved = (stage_file.parent / reference).resolve()
                try:
                    relative = resolved.relative_to(root)
                except ValueError:
                    continue  # outside the snapshot
                parent = str(relative.parent)
                if not parent.startswith(f"{UPSTREAM_PROPS_DIR}/"):
                    continue
                if resolved.is_file():
                    continue  # already downloaded with an earlier round
                extra.setdefault(parent, set()).add(f"{parent}/**")
    return {prop_dir: tuple(sorted(patterns)) for prop_dir, patterns in extra.items()}




# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source", type=Path, default=None,
        help="Existing dataset snapshot (with Props/ and the CSV).  If "
             f"omitted, the CSV catalog is fetched from {HF_DATASET_REPO}, "
             "the selected props are resolved from it, and each selected "
             "prop subtree (the few MB under its Props/…/ directory — not "
             "the whole 14.45 GB dataset) is downloaded into <output-dir>/ "
             "and repackaged from there.",
    )
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="Path to the dataset's CSV catalog (default: <source>/ "
             f"{CSV_NAME}, or a Hub download into <output-dir>/).  Without "
             "--source only the CSV is fetched before selection so the "
             "per-subtree downloads only cover the selected props.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to place the prepared 'objects/simready_warehouse' tree in.",
    )
    parser.add_argument(
        "--categories", default=",".join(DEFAULT_CATEGORIES),
        help="Comma-separated subset of CSV classification classes to "
             f"ingest: {', '.join(ALL_KNOWN_CATEGORIES)} (default: "
             f"{', '.join(DEFAULT_CATEGORIES)}; 475 'Prop general', 175 "
             "'hand manipulation', 103 'Assembly' rows).",
    )
    parser.add_argument(
        "--objects", type=str, default=None,
        help="Comma-separated subset of prop ids or CSV asset names "
             "(substring match) to ingest (default: all in the selected "
             "categories).",
    )
    parser.add_argument(
        "--skip-gate", action="store_true",
        help="Skip the Genesis GPU sanity-load gate (loader + bbox + render).",
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="Skip the per-prop thumbnail render (the gate still checks loads + bboxes).",
    )
    parser.add_argument(
        "--render-dir", type=Path, default=Path("/tmp/simready_warehouse_sanity"),
        help="Directory for the per-prop eyeball renders "
             "(default /tmp/simready_warehouse_sanity).",
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
    args = parser.parse_args(argv)

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    unknown = [c for c in categories if c not in ALL_KNOWN_CATEGORIES]
    if unknown:
        raise SystemExit(
            f"Unknown SimReady Warehouse category/ies: {unknown}. Known: "
            f"{list(ALL_KNOWN_CATEGORIES)} (CSV classes: Prop general 475, "
            "Prop general hand manipulation 175, Assembly 103)."
        )

    source: Path
    csv_file: Path
    if args.source is not None:
        source = args.source.expanduser().resolve()
        if not (source / UPSTREAM_PROPS_DIR).is_dir():
            raise SystemExit(
                f"No '{UPSTREAM_PROPS_DIR}/' directory in --source: {source}"
            )
        csv_file = (args.csv or source / CSV_NAME).expanduser().resolve()
        if not csv_file.is_file():
            raise SystemExit(f"No CSV catalog at {csv_file} (pass --csv).")
    else:
        # No local snapshot: fetch the CSV catalog, resolve the selection,
        # then download each selected prop's subtree (its `Props/…/<dir>/`
        # directory — a few MB each, not the whole 14.45 GB dataset) into
        # <output-dir>/, materialising the upstream relative layout.  The
        # packaging below then reads from that shadow snapshot.
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.csv is not None:
            csv_file = args.csv.expanduser().resolve()
            if not csv_file.is_file():
                raise SystemExit(f"No CSV catalog at {csv_file}.")
        else:
            print(f"Fetching the {HF_DATASET_REPO} CSV catalog ...")
            csv_file = _fetch_catalog(output_dir)
            print(f"  -> {csv_file}")
        rows = read_csv_catalog(csv_file)
        selected = select_records(rows, categories, objects=(
            [o for o in args.objects.split(",") if o.strip()] if args.objects else None
        ))
        selected_ids = {object_id for object_id, _record in selected}
        relative_paths = [
            row["relative_path"] for row in rows
            if _csv_object_id(row["relative_path"]) in selected_ids
        ]
        source = fetch_dataset(output_dir, relative_paths=relative_paths)

    out_root, _kept = prepare(
        source,
        args.output_dir.expanduser().resolve(),
        categories,
        csv_file,
        objects=(
            [o for o in args.objects.split(",") if o.strip()] if args.objects else None
        ),
        gate=not args.skip_gate,
        bbox_tol_abs=args.bbox_tol_abs,
        bbox_tol_rel=args.bbox_tol_rel,
        render_dir=None if args.no_render else args.render_dir,
    )

    print(
        "\nTo upload (requires write access to the org):\n"
        f"  hf upload {HF_UPLOAD_REPO} \\\n"
        f"      {out_root} {OUTPUT_ASSETS_SUBDIR} --repo-type dataset\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
