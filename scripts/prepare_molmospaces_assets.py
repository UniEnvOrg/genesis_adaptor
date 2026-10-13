#!/usr/bin/env python3
# Upstream asset license: CC BY 4.0 for the isaac/objects/thor subset (per the
# allenai/molmospaces dataset card; Objaverse-derived items are ODC-BY 1.0).
# This packager only rehosts the USD payloads unchanged; the original license
# applies to the packaged assets.  Attribution required.
"""Prepare MolmoSpaces (Ai2-THOR) USD furniture assets for the collection's assets repo.

Downloads (or reuses) the `allenai/molmospaces
<https://huggingface.co/datasets/allenai/molmospaces>`_ dataset's
``isaac/objects/thor`` USD subset and repackages it into the layout expected
by ``unienv_genesis_collection.objects.hub_objects.articulated.molmospaces_furniture``
(one self-contained directory per object, loaded with ``gs.morphs.USD`` via
``scene.add_stage``)::

    <output>/objects/molmospaces/LICENSE                  # CC BY 4.0 attribution
    <output>/objects/molmospaces/README.md                # provenance note
    <output>/objects/molmospaces/catalog.json             # generated annotations + tags
    <output>/objects/molmospaces/packaging_log.json       # per-object gate log
    <output>/objects/molmospaces/<object_id>/<object_id>.usda
    <output>/objects/molmospaces/<object_id>/Textures/... # relative textures

**Joint-body swap (the critical converter).**  MolmoSpaces authors every
UsdPhysics joint with ``body0`` = child and ``body1`` = parent — the reverse
of Genesis's loader contract.  As shipped, 12/18 probed samples fail to load
(``Link … has multiple parents``) and the remaining 6 load with the movable
part in the base link.  Every packaged stage is therefore rewritten with
:func:`~unienv_genesis_collection.utils.usd_conversions.swap_usd_joint_bodies`
(relationship targets AND the per-side ``localPos0/1``/``localRot0/1``
frames); the sanity gate below verifies the *swapped* stage in Genesis
(single entity, correct DOF kinds, closed + halfway-opened bbox).

**Variant choice.**  Each upstream object ships three variants: ``<Name>``
(primitive colliders), ``<Name>_prim`` (same, doc-string differs) and
``<Name>_mesh`` (convex-hull ``Mesh`` colliders — e.g. 14 colliders for
Fridge_26 instead of 281).  The **``_mesh`` variant is packaged** when it
exists: closest collider proxy at the lowest collider count (falls back to
the base variant when absent).

Every packaged object must pass a **Genesis sanity gate**: loader acceptance
through ``MolmoSpacesFurniture.load_into_scene`` with a single-entity assert,
a non-empty moving (non-fixed) joint set, unique link leaf names, the offline
USD FK union bbox vs ``entity.get_vAABB()`` at q = 0 *and* at a halfway-opened
configuration, plus a CUDA-only thumbnail render.  Failing objects are
dropped with a warning instead of failing the run.

License: the whole ``isaac/objects/thor`` subset is **CC BY 4.0** per the
dataset card ("The Objaverse subset is licensed under ODC-BY 1.0. All other
data subsets are licensed under CC BY 4.0"); the Objaverse subset (ODC-BY
with per-object NC entries) is **never** downloaded — only ``thor`` archives
are fetched, so nothing non-CC-BY-4.0 enters the pack.

Usage::

    python scripts/prepare_molmospaces_assets.py --output-dir /tmp/molmo_assets
    # or with an already-extracted snapshot (per-object variant dirs):
    python scripts/prepare_molmospaces_assets.py --source <dir> \\
        --output-dir /tmp/molmo_assets --objects Fridge_26,IKEACabinet_1

Then upload to the Hub (needs write access to the org)::

    hf upload UniEnvOrg/unienv_genesis_collection_assets \\
        /tmp/molmo_assets/objects/molmospaces objects/molmospaces --repo-type dataset
"""

from __future__ import annotations

import argparse
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
    sanity_check_articulated,
)
from _packaging.usd_verify import verify_object_assets as _verify_object_assets  # noqa: E402
sanity_check_object = sanity_check_articulated  # the pre-refactor family name



HF_DATASET_REPO = "allenai/molmospaces"
HF_OBJECTS_PREFIX = "isaac/objects/thor/20260128"
OUTPUT_ASSETS_SUBDIR = "objects/molmospaces"
HF_UPLOAD_REPO = "UniEnvOrg/unienv_genesis_collection_assets"

# THOR object families (the articulated ones per ``usd_assets_metadata.json``:
# 189/382 USD objects are articulated).  ``--categories`` keys into these.
FAMILY_TEMPLATES: Dict[str, Dict[str, object]] = {
    "fridge": dict(kind="appliance", rooms=("kitchen",), detail="a refrigerator"),
    "microwave": dict(kind="appliance", rooms=("kitchen",), detail="a microwave"),
    "toaster": dict(kind="appliance", rooms=("kitchen",), detail="a toaster"),
    "laptop": dict(kind="appliance", rooms=("office",), detail="a laptop"),
    "box": dict(kind="household item", rooms=("utility",), detail="a box with a hinged lid"),
    "book": dict(kind="household item", rooms=(), detail="a book"),
    "cabinet": dict(kind="furniture", rooms=("kitchen",), detail="an IKEA-style cabinet"),
    "side_table": dict(kind="furniture", rooms=("living_room",), detail="a side table"),
    "laundry_hamper": dict(kind="household item", rooms=("laundry",), detail="a laundry hamper"),
    "dresser": dict(kind="furniture", rooms=("bedroom",), detail="a dresser"),
    "desk": dict(kind="furniture", rooms=("office",), detail="a desk"),
    "bathroom_faucet": dict(kind="fixture", rooms=("bathroom",), detail="a bathroom faucet"),
    "doorway": dict(kind="furniture", rooms=("interior",), detail="a doorway"),
    "toilet": dict(kind="fixture", rooms=("bathroom",), detail="a toilet"),
    "safe": dict(kind="furniture", rooms=("office",), detail="a safe"),
    "shelving_unit": dict(kind="furniture", rooms=("living_room",), detail="a shelving unit"),
}
DEFAULT_CATEGORIES: Tuple[str, ...] = (
    "fridge", "microwave", "toaster", "laptop", "box", "cabinet", "side_table",
)
ALL_KNOWN_CATEGORIES: Tuple[str, ...] = tuple(sorted(FAMILY_TEMPLATES))

# Bbox cross-check tolerances (offline-FK box proxy vs real mesh geometry)
# — defined once in ``_packaging.sanity`` (BBOX_TOL_ABS/BBOX_TOL_REL) and
# imported above.

LICENSE_TEXT = """\
MolmoSpaces Ai2-THOR USD object assets.

Dataset: "MolmoSpaces" — allenai/molmospaces,
https://huggingface.co/datasets/allenai/molmospaces

The `isaac/objects/thor` USD subset packaged here is licensed under
**CC BY 4.0** (Creative Commons Attribution 4.0 International,
https://creativecommons.org/licenses/by/4.0/legalcode.en): per the dataset
card, "The Objaverse subset is licensed under ODC-BY 1.0. All other data
subsets are licensed under CC BY 4.0."  Only the thor subset is packaged —
the Objaverse subset (ODC-BY, with per-object non-commercial entries) is
never fetched.

Attribution: "MolmoSpaces THOR objects (Model(s) by the Allen Institute for
AI (Ai2)), licensed under CC BY 4.0."  The dataset is intended for research
and educational use in accordance with Ai2's Responsible Use Guidelines
(https://allenai.org/responsible-use).

Repackaged by genesis_adaptor/scripts/prepare_molmospaces_assets.py.
"""

PROVENANCE_README = """\
# MolmoSpaces (Ai2-THOR) articulated USD furniture

Repackaged from the [MolmoSpaces dataset](https://huggingface.co/datasets/allenai/molmospaces)
(Isaac-compatible USD objects converted from Ai2-THOR, CC BY 4.0): one
self-contained directory per object under `objects/molmospaces/<object_id>/`,
containing the root USD stage (`<object_id>.usda`, renamed from the upstream
`<Name>_mesh.usda`) plus its `Textures/` sidecar (relative references).

* **The `<Name>_mesh` variant is packaged** (convex-hull mesh colliders —
  closest proxy at the lowest collider count; the base/`_prim` variants
  author primitive colliders instead).
* **Every packaged stage carries the mandatory joint-body0/body1 swap:**
  MolmoSpaces authors `body0` = child / `body1` = parent (MJCF order), the
  reverse of Genesis's USD loader contract — the packaged stages swap the
  relationship targets *and* the per-side `localPos0/1`/`localRot0/1`
  frames (`unienv_genesis_collection.utils.usd_conversions`), verified
  per object in the packaging gate (per-object details in
  `packaging_log.json`).
* Materials are already UsdPreviewSurface (no MDL in the thor USD subset).
* Genesis grants USD joints by **full prim path**; the descriptor exposes
  *clean*, leaf-prim joint/link names (see the Articulated Rooms docs).
* Every shipped object passed the Genesis sanity gate: single-entity
  `add_stage` load, non-empty moving-joint set, the offline box-proxy FK
  union bbox cross-checked against `entity.get_vAABB()` at the closed *and*
  a halfway-opened configuration.

Only the **thor** subset (CC BY 4.0) is packaged; the Objaverse subset is
never fetched — see `LICENSE`.

When using these models, please attribute the MolmoSpaces dataset
("MolmoSpaces THOR objects, Model(s) by the Allen Institute for AI (Ai2),
CC BY 4.0") — see `LICENSE`.
"""


# ---------------------------------------------------------------------------
# Upstream discovery
# ---------------------------------------------------------------------------

def family_of(upstream_id: str) -> str:
    """The THOR family of an upstream object id (``Fridge_26`` → ``fridge``).

    Matched against the template keys with separators removed, so camel
    names (`IKEACabinet_1`) hit ``cabinet`` and snake names
    (``Side_Table_302_1``) hit ``side_table``.  Longest key wins; an unknown
    family falls back to the stripped token itself (snake-cased).
    """
    token = upstream_id.rstrip("_0123456789")
    compact = token.lower().replace("_", "")
    best = None
    for family in FAMILY_TEMPLATES:
        if family.replace("_", "") in compact:
            if best is None or len(family) > len(best):
                best = family
    if best is not None:
        return best
    # Unknown family: snake-case the leading word for the generic template.
    return "_".join(word.lower() for word in token.split("_"))


def list_molmo_objects(source: Path, categories: Sequence[str],
                       articulated: Optional[Dict[str, bool]] = None) -> List[Tuple[str, Path]]:
    """List ``(packaged_object_id, upstream_variant_dir)`` pairs on *source*.

    *source* is a tree of extracted upstream object dirs (each containing
    the three variants ``<Name>/``, ``<Name>_prim/``, ``<Name>_mesh/`` — or
    just one of them).  The packaged id is the lowercased upstream id with
    camel-cases split (``IKEACabinet_1`` → ``ikeacabinet_1``); the
    **``_mesh`` variant is preferred** (convex-hull colliders), falling back
    to ``_prim`` and the base variant.  When *articulated* is given (the
    upstream ``usd_assets_metadata.json`` index), non-articulated objects
    are skipped — the family contract needs a moving joint set.

    Group archives extract to a directory that holds nothing but the
    extraction marker: ``thor_Cabinet.tar.zst``'s real members are the
    *sibling* dirs ``Cabinet_Corner_01/``, …, so ``extract_root/Cabinet``
    contains only ``.extracted``.  Such a stage-less directory is skipped
    (it can never repackage — ``repackage_object`` would raise
    ``FileNotFoundError`` on the missing ``*.usda``), which keeps the
    ``Cabinet``/``DecorativeBox``/``Side_Table_20x…`` phantom ids out of the
    object list entirely.
    """
    found: Dict[str, Path] = {}
    for variant_dir in sorted(source.iterdir()):
        if not variant_dir.is_dir():
            continue
        if not any(variant_dir.rglob("*.usda")):
            # No stage anywhere in the directory: a marker-only group-archive
            # dir (or an otherwise empty extraction) — not an object.
            continue
        variant = variant_dir.name
        if variant.endswith("_mesh"):
            base = variant[: -len("_mesh")]
            rank = 0
        elif variant.endswith("_prim"):
            base = variant[: -len("_prim")]
            rank = 1
        else:
            base = variant
            rank = 2
        if base in found and rank >= _variant_rank(found[base]):
            continue  # a preferred variant already claimed this object
        found[base] = variant_dir

    pairs = []
    for base, variant_dir in sorted(found.items()):
        family = family_of(base)
        if family not in categories:
            continue
        if articulated is not None and articulated.get(base) is False:
            continue
        flat_id = base.lower()
        pairs.append((flat_id, variant_dir))
    return sorted(pairs)


def _variant_rank(variant_dir: Optional[Path]) -> int:
    if variant_dir is None:
        return 3
    name = variant_dir.name
    if name.endswith("_mesh"):
        return 0
    if name.endswith("_prim"):
        return 1
    return 2


# ---------------------------------------------------------------------------
# Repackaging (variant pick + joint-body swap)
# ---------------------------------------------------------------------------

def repackage_object(variant_dir: Path, object_id: str, out_root: Path,
                     swapped: bool = True) -> Tuple[Path, int]:
    """Repackage one upstream variant dir into ``out_root/<object_id>/``.

    Copies the variant tree verbatim (the ``<Name>.usda`` stage + its
    ``Textures/`` sidecar), renames the root layer to ``<object_id>.usda``
    (relative texture references are directory-based, so the rename is safe)
    and applies the **joint-body swap** (MolmoSpaces authors the inverted
    order; Genesis needs the opposite).  Verifies that every file the pxr
    layer stack + texture references name exists in the packaged directory.

    Returns:
        ``(packaged_dir, n_swapped_joints)``.
    """
    stage_files = sorted(p for p in variant_dir.glob("*.usda") if p.is_file())
    if len(stage_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one .usda stage in {variant_dir}, found "
            f"{[p.name for p in stage_files]}."
        )
    stage_file = stage_files[0]

    obj_out_dir = out_root / object_id
    obj_out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(variant_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(variant_dir)
        (obj_out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, obj_out_dir / rel)
    packaged_stage = obj_out_dir / f"{object_id}.usda"
    if stage_file.name != packaged_stage.name:
        (obj_out_dir / stage_file.name).rename(packaged_stage)

    n_swapped = 0
    if swapped:
        from unienv_genesis_collection.utils.usd_conversions import swap_usd_joint_bodies

        n_swapped = swap_usd_joint_bodies(packaged_stage)

    _verify_object_assets(packaged_stage, obj_out_dir)
    return obj_out_dir, n_swapped



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
    the same object rewrites its entry.
    """
    return _write_catalog_impl(
        out_root, records, merge=merge, drop=drop, overrides=CATALOG_OVERRIDES,
    )

# Hand fixes (catalog entries) — applied after the generated entries.
CATALOG_OVERRIDES: Dict[str, dict] = {}


def annotate_object(upstream_id: str, n_revolute: int, n_prismatic: int) -> Tuple[str, dict]:
    """One catalog entry from the family template + the gate's joint counts.

    Annotations broad → detailed: ``[kind, "a <family>", "<the joint-summary
    line>"]``; room-hint tags come from the template's ``rooms`` heuristic
    and the ``articulated`` tag is added by the caller.
    """
    family = family_of(upstream_id)
    template = FAMILY_TEMPLATES.get(family, dict(kind="object", rooms=()))
    kind = str(template["kind"])
    words = family.replace("_", " ")
    specifics = []
    if n_revolute:
        if family in ("fridge", "microwave", "box", "cabinet", "doorway", "dresser",
                      "shelving_unit", "safe", "desk"):
            specifics.append(
                f"{n_revolute} hinged door{'' if n_revolute == 1 else 's'}"
            )
        else:
            specifics.append(
                f"{n_revolute} hinged part{'' if n_revolute == 1 else 's'}"
            )
    if n_prismatic:
        specifics.append(
            f"{n_prismatic} sliding part{'' if n_prismatic == 1 else 's'}"
        )
    annotations = [
        kind,
        f"a {words}",
        (
            f"a {words} with {', '.join(specifics)}"
            if specifics
            else f"a plain {words} variant"
        ),
    ]
    entry_tags = {kind.replace(" ", "_"), family}
    entry_tags.update(tuple(template["rooms"]))  # type: ignore[arg-type]
    entry = {
        "language_annotations": list(dict.fromkeys(annotations)),
        "tags": sorted(entry_tags),
    }
    return kind, entry


# ---------------------------------------------------------------------------
# Genesis sanity gate (GPU) — the shared loop (``_packaging.sanity``) wired to
# this family's descriptor factory
# ---------------------------------------------------------------------------

def _gate_descriptor(object_id: str):
    """Build the family descriptor for one packaged id (the only per-family
    part of the gate).  Imported lazily so the asset repointing
    (``asset_path.ASSETS_DIR`` → *output_dir*) is already in place."""
    from unienv_genesis_collection.objects.hub_objects.articulated.molmospaces_furniture import (
        MolmoSpacesFurniture,
    )

    return MolmoSpacesFurniture(object_id)


def run_sanity_gate(
    output_dir: Path,
    object_ids: Sequence[str],
    render_dir: Optional[Path] = None,
    bbox_tol_abs: float = BBOX_TOL_ABS,
    bbox_tol_rel: float = BBOX_TOL_REL,
) -> Tuple[List[str], List[dict], List[dict]]:
    """Sanity-load every packaged object; drop (warn on) the failing ones.

    Runs from the freshly packaged tree only (a Hub download during
    packaging would mask an incomplete repackage — assets are repointed at
    *output_dir* and downloads are stubbed out).  Returns
    ``(kept_object_ids, passed_records, dropped_records)``.
    """
    return _run_sanity_gate_impl(
        output_dir, object_ids,
        descriptor_factory=_gate_descriptor,
        check=sanity_check_articulated,
        render_dir=render_dir,
        bbox_tol_abs=bbox_tol_abs,
        bbox_tol_rel=bbox_tol_rel,
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
    articulated: Optional[Dict[str, bool]] = None,
) -> Tuple[Path, List[str]]:
    """Package the selected families into ``output_dir/objects/molmospaces``.

    Returns ``(out_root, kept_object_ids)``.
    """
    out_root = output_dir / OUTPUT_ASSETS_SUBDIR
    out_root.mkdir(parents=True, exist_ok=True)

    pairs = list_molmo_objects(source, categories, articulated)
    if objects:
        wanted = {o.strip().lower() for o in objects if o.strip()}
        selected = [
            (object_id, variant_dir)
            for object_id, variant_dir in pairs
            if (object_id.lower() in wanted
                or variant_dir.name.lower() in wanted
                or any(w in object_id.lower() or w in variant_dir.name.lower() for w in wanted))
        ]
        if not selected:
            raise SystemExit(
                f"--objects matched none of the {len(pairs)} upstream objects."
            )
    else:
        selected = pairs
    if not selected:
        raise SystemExit(
            f"No MolmoSpaces objects found in {source} for {list(categories)}."
        )

    upstream_ids: Dict[str, str] = {}
    packaged: List[str] = []
    swap_counts: Dict[str, int] = {}
    print(f"Repackaging {len(selected)} object(s) ...")
    for object_id, variant_dir in selected:
        try:
            _dir, n_swapped = repackage_object(variant_dir, object_id, out_root)
        except Exception as exc:  # noqa: BLE001 — repackage failure ⇒ drop
            print(f"  [repackage] {object_id}: DROPPED — {type(exc).__name__}: {exc}")
            shutil.rmtree(out_root / object_id, ignore_errors=True)
            continue
        upstream_ids[object_id] = variant_dir.name
        packaged.append(object_id)
        swap_counts[object_id] = n_swapped
        print(f"  -> {out_root / object_id} (swapped {n_swapped} joint(s))")

    # Preliminary catalog (template annotations) so the gate's descriptors
    # resolve offline; rewritten with the gate survivors below.
    prelim_entries = [
        dict(object_id=object_id, catalog_entry=annotate_object(
            upstream_ids[object_id], 0, 0)[1])
        for object_id in packaged
    ]
    write_catalog(out_root, prelim_entries)

    if gate:
        print(f"\nGenesis sanity gate on {len(packaged)} object(s) ...")
        kept, passed_records, dropped_records = _run_sanity_gate_batched(
            __file__, run_sanity_gate, output_dir, packaged,
            render_dir=render_dir,
            bbox_tol_abs=bbox_tol_abs,
            bbox_tol_rel=bbox_tol_rel,
        )
        gate_entries: List[dict] = []
        for record in passed_records:
            record["n_files"] = len(list((out_root / record["object_id"]).rglob("*")))
            record["n_swapped_joints"] = swap_counts[record["object_id"]]
            entry = annotate_object(
                upstream_ids[record["object_id"]],
                sum(1 for j in record["joints"] if j["type"] == "REVOLUTE"),
                sum(1 for j in record["joints"] if j["type"] == "PRISMATIC"),
            )[1]
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
# Download (HF shard, full download + extract)
# ---------------------------------------------------------------------------

def _load_articulated_index(source_dir: Path) -> Dict[str, bool]:
    """``usd_assets_metadata.json`` (1667 assets) → ``{id: articulated}``."""
    candidates = [
        path
        for path in sorted(source_dir.rglob("usd_assets_metadata.json"))
        if path.is_file()
    ]
    if not candidates:
        return {}
    index = json.loads(candidates[0].read_text())
    return {
        entry.get("asset_id", key): bool(entry.get("articulated", False))
        for key, entry in index.items()
    }


def fetch_dataset(output_dir: Path, categories: Sequence[str]) -> Tuple[Path, Dict[str, bool]]:
    """Download the thor object subset from the HF dataset and extract it.

    The subset is stored as a single ~1.1 GB tar shard whose members are
    per-object ``.tar.zst`` archives (``arrow_table.json`` records their
    offsets — a full download + extract keeps the tooling simple).  Only
    ``thor`` files are fetched: the whole subset is CC BY 4.0, so nothing
    outside that license can enter the pack.
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    # Offset table + the object shard.  The articulated-flag index is fetched
    # best-effort below — upstream does not currently ship a thor-scoped
    # ``usd_assets_metadata.json``, so its absence must not abort the run
    # (``_load_articulated_index`` then returns {} and articulation is decided
    # by the sanity gate's joint set).
    snapshot_download(
        HF_DATASET_REPO,
        repo_type="dataset",
        allow_patterns=[
            f"{HF_OBJECTS_PREFIX}/arrow_table.json",
            f"{HF_OBJECTS_PREFIX}/shards/00000.tar",
        ],
    )
    import tarfile

    try:
        import zstandard  # noqa: F401
    except ImportError as exc:  # noqa: PERF203
        raise SystemExit(
            "The MolmoSpaces shard stores per-object .tar.zst archives — "
            "`pip install zstandard` and re-run."
        ) from exc
    import io
    import zstandard

    table_file = hf_hub_download(
        HF_DATASET_REPO, f"{HF_OBJECTS_PREFIX}/arrow_table.json", repo_type="dataset",
    )
    rows = json.loads(Path(table_file).read_text())
    if isinstance(rows, dict):  # HF arrow-table column layout
        keys = list(next(iter(rows.values())))
        rows = [dict(zip(rows, (rows[col][i] for col in rows))) for i in range(len(keys))]

    shard_file = hf_hub_download(
        HF_DATASET_REPO, f"{HF_OBJECTS_PREFIX}/shards/00000.tar", repo_type="dataset",
    )
    # Articulation is decided by the sanity gate's joint set (the upstream
    # ``usd_assets_metadata.json`` index is not fetched — its per-object
    # archive shares its name with object archives and is best left out of
    # the object tree).  ``_load_articulated_index`` below returns {} here,
    # so the caller falls back to the gate.

    extract_root = Path(output_dir) / "_molmo_extract"
    extract_root.mkdir(parents=True, exist_ok=True)
    wanted_ids = {obj.strip().lower() for obj in (_CLI_OBJECTS or [])}
    with tarfile.open(shard_file) as shard:
        # The shard is an *uncompressed* tar, so ``shard.fileobj`` is the raw
        # underlying file and ``arrow_table`` offsets index into it directly.
        # (``TarFile`` itself has no random ``seek``/``read``.)
        raw = shard.fileobj
        for row in rows:
            archive_name = row["path"]
            base = archive_name.removesuffix(".tar.zst")
            if not base.startswith("thor_"):
                continue  # the non-thor subsets are never touched
            base = base[len("thor_"):]
            # The articulated-flag index archive holds a lone
            # ``usd_assets_metadata.json`` (``removesuffix(".tar.zst")`` leaves
            # the ``.json``), not an object — skip it so it never lands in the
            # object tree (its members would collide with same-named paths).
            if base in ("usd_assets_metadata", "usd_assets_metadata.json"):
                continue
            if wanted_ids and not any(w in base.lower() for w in wanted_ids):
                continue
            object_dir = extract_root / base
            marker = object_dir / ".extracted"
            if marker.is_file():
                continue
            raw.seek(row["offset"])
            payload = raw.read(row["size"])
            if len(payload) != row["size"]:
                raise SystemExit(
                    f"Short read for {archive_name}: got {len(payload)} of "
                    f"{row['size']} bytes (truncated shard download?)."
                )
            # Each archive's members are the variant dirs themselves —
            # ``<Base>/<Base>.usda``, ``<Base>_mesh/<Base>_mesh.usda``,
            # ``<Base>_prim/<Base>_prim.usda`` (+ a ``Textures/`` sidecar per
            # variant) — with NO single top-level wrapper.  Extract verbatim
            # into ``extract_root`` so the layout matches what
            # ``list_molmo_objects`` scans.
            object_dir.mkdir(parents=True, exist_ok=True)
            with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(payload)) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as tar:
                    for member in tar:
                        relative = _safe_member_path(member.name)
                        if relative is None:
                            continue
                        target = extract_root / relative
                        if member.isdir():
                            target.mkdir(parents=True, exist_ok=True)
                        elif member.isreg():
                            target.parent.mkdir(parents=True, exist_ok=True)
                            if target.is_dir():
                                # A stale directory from an earlier (buggy)
                                # run occupies this file's path — remove it.
                                shutil.rmtree(target)
                            with tar.extractfile(member) as src, open(target, "wb") as dst:
                                shutil.copyfileobj(src, dst)
            marker.write_text("ok\n")
            print(f"  extracted {base} ({row['size'] / 1e6:.1f} MB)")
    # No articulated-flag index was fetched (see the note above); the gate
    # decides articulation, so this returns {}.
    return extract_root, _load_articulated_index(extract_root)


def _safe_member_path(member_name: str) -> Optional[str]:
    """Return a normalised relative member path, or ``None`` to skip it.

    Guards against absolute paths and ``..`` escapes (``filter='data'`` is
    unavailable on a streaming ``r|`` tar, so the check is done here).
    """
    relative = Path(member_name)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    cleaned = Path(*[p for p in relative.parts if p not in ("", ".")])
    if not cleaned.parts:
        return None
    return str(cleaned)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source", type=Path, default=None,
        help="Existing tree of extracted MolmoSpaces object variant dirs "
             "(each dir is one <Name>, <Name>_prim or <Name>_mesh variant). "
             "If omitted, the thor subset is fetched from "
             f"{HF_DATASET_REPO} (a ~1.1 GB shard + extract).",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to place the prepared 'objects/molmospaces' tree in.",
    )
    parser.add_argument(
        "--categories", default=",".join(DEFAULT_CATEGORIES),
        help="Comma-separated subset of THOR object families to package "
             f"(default: {', '.join(DEFAULT_CATEGORIES)}; known: "
             f"{', '.join(ALL_KNOWN_CATEGORIES)}).",
    )
    parser.add_argument(
        "--objects", type=str, default=None,
        help="Comma-separated subset of upstream object ids (substring "
             "match) to ingest (default: all in the selected categories).",
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
        "--render-dir", type=Path, default=Path("/tmp/molmospaces_sanity"),
        help="Directory for the per-object eyeball renders "
             "(default /tmp/molmospaces_sanity).",
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
        "--include-static", action="store_true",
        help="Also package upstream objects whose metadata flags them "
             "non-articulated (the gate still drops them for an empty "
             "moving-joint set).",
    )
    args = parser.parse_args(argv)

    global _CLI_OBJECTS
    _CLI_OBJECTS = [o for o in (args.objects or "").split(",") if o.strip()]

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    unknown = [c for c in categories if c not in ALL_KNOWN_CATEGORIES]
    if unknown:
        raise SystemExit(
            f"Unknown MolmoSpaces category/ies: {unknown}. Known: "
            f"{list(ALL_KNOWN_CATEGORIES)}."
        )

    articulated: Optional[Dict[str, bool]] = None
    if args.source is not None:
        source = args.source.expanduser().resolve()
        if not source.is_dir():
            raise SystemExit(f"--source is not a directory: {source}")
    else:
        print(f"Fetching {HF_DATASET_REPO} ({', '.join(categories)}) via huggingface_hub ...")
        source, articulated = fetch_dataset(
            args.output_dir.expanduser().resolve(), categories,
        )
        print(f"  -> {source}")
    if not args.include_static:
        articulated = articulated or _load_articulated_index(source)
        if not articulated:
            print(
                "  [note] no usd_assets_metadata.json found next to --source: "
                "articulation is decided by the sanity gate's joint set only."
            )

    out_root, _kept = prepare(
        source,
        args.output_dir.expanduser().resolve(),
        categories,
        objects=(_CLI_OBJECTS or None),
        gate=not args.skip_gate,
        bbox_tol_abs=args.bbox_tol_abs,
        bbox_tol_rel=args.bbox_tol_rel,
        render_dir=None if args.no_render else args.render_dir,
        articulated=articulated,
    )

    print(
        "\nTo upload (requires write access to the org):\n"
        f"  hf upload {HF_UPLOAD_REPO} \\\n"
        f"      {out_root} {OUTPUT_ASSETS_SUBDIR} --repo-type dataset\n"
    )
    return 0


_CLI_OBJECTS: List[str] = []


if __name__ == "__main__":
    sys.exit(main())
