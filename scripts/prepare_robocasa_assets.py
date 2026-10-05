#!/usr/bin/env python3
"""Prepare RoboCasa kitchen fixture assets for the collection's assets repo.

RoboCasa (https://robocasa.ai, CC BY 4.0) ships its kitchen fixtures as MJCF
models.  This script repackages the fixture registry subset
(``cabinet_hinge``, ``cabinet_single``, ``drawer``, ``panel``, handles,
``dishwasher``, ``oven``, ``fridge`` ×3, ``microwave``, ``stove``, ``sink``,
``counters``) into one self-contained directory per fixture, the layout
expected by ``unienv_genesis_collection.objects.hub_objects.RoboCasaFixture``
(MJCF ``model.xml`` loaded via ``gs.morphs.MJCF``; jointless root = welded
base, fixed)::

    <output>/objects/robocasa/LICENSE                 # CC BY 4.0 attribution
    <output>/objects/robocasa/README.md               # provenance
    <output>/objects/robocasa/catalog.json            # annotations + tags
    <output>/objects/robocasa/<fixture_id>/model.xml
    <output>/objects/robocasa/<fixture_id>/visuals/*.obj (+ *.png)
    <output>/objects/robocasa/<fixture_id>/meshes/*.obj
    <output>/objects/robocasa/packaging_log.json      # per-fixture gate log

plus the kitchen arena shells for the room builder::

    <output>/scenes/rooms/robocasa_kitchen/<variant>/model.xml (+ textures)

Sources (``--source``, else Hugging Face):

* a local RoboCasa assets download — the ``robocasa/models/assets`` tree
  produced by ``python -m robocasa.scripts.download_kitchen_assets --type all``
  (or any directory holding ``fixtures/`` + ``textures/``, e.g. the extracted
  ``fixtures.zip`` / ``textures.zip`` of the Hub mirrors);
* otherwise ``huggingface_hub.snapshot_download`` of ``--repo-id``, whose
  ``fixtures.zip`` + ``textures.zip`` are extracted into a scratch directory.

.. note::
   The *official* ``robocasa/robocasa-assets`` Hub repo only publishes the
   ``accessories`` fixture families in its ``fixtures.zip`` — its appliance
   families (dishwashers, fridges, ovens, stoves, sinks, microwaves) are
   distributed through Box links whose direct download requires a
   browser-like user agent and which are documented as the alternate source
   (``python -m robocasa.scripts.download_kitchen_assets``).  The default repo
   is therefore the community mirror ``jianzhang96/robocasa-assets``, whose
   ``fixtures.zip`` + ``textures.zip`` match the ``download_kitchen_assets``
   layout and include every kitchen fixture family.

Two kinds of fixtures are packaged:

* **Static** fixtures (appliances, counter modules): authored ``model.xml`` +
  ``visuals/`` meshes/textures, copied with every referenced file — including
  recursively resolved ``<include>``s, mirroring
  ``genesis.utils.mjcf.build_model`` and the collection's
  ``_mjcf_required_files`` — with references rewritten directory-local and any
  root ``<freejoint/>`` stripped (fixed furniture).  Authored collision geoms
  are used as-is.
* **Composed** fixtures (cabinets, drawer, panel, handles): upstream RoboCasa
  composes these procedurally at runtime from primitive templates
  (``robocasa/models/fixtures/{cabinets,cabinet_panels,handles}.py``); no
  static MJCF exists.  This script ports that composition for a fixed style
  (slab panels, bar/boxed/knob handles, wood texture) and bakes the composed
  MuJoCo box/cylinder primitives into exact-dimension **mesh geoms**, because
  the collection's offline MJCF forward-kinematics walker
  (``ArticulatedObject`` → union bbox, plan §1.3) only reads ``<mesh>``
  geoms — a box-geom-only fixture has no offline visual geometry at all.  The
  baked meshes keep both the offline FK bbox and the packaging-time
  ``get_vAABB()`` cross-check meaningful; collision becomes a convex
  box/cylinder mesh (identical silhouette) and the reset-region markers stay
  as authored.

Every packaged fixture must pass a **Genesis sanity gate** (plan risk #1):
loaded in a throwaway GPU scene through ``RoboCasaFixture.load_into_scene``,
non-empty joint set where the registry expects articulation, and the offline
MJCF-FK union bbox cross-checked against ``entity.get_vAABB()`` at q = 0
within tolerance.  Survivors also get a standing thumbnail render into
``--render-dir`` (recorded as ``record["render"]`` in the packaging log;
skipped on the CPU backend, where Genesis's Rasterizer camera does not work —
a render failure is a warning, never a drop).  The discovered joint/link
names are recorded in the packaging log (informational only — nothing
per-asset is baked into the catalog).  Fixtures that fail are dropped with a
warning, not a pipeline failure.  ``cabinet_hinge`` and ``drawer`` are gated
first (cheapest signal).

Usage::

    python scripts/prepare_robocasa_assets.py --output-dir /tmp/robocasa_assets
    python scripts/prepare_robocasa_assets.py --source ~/robocasa/robocasa/models/assets \\
        --output-dir /tmp/robocasa_assets
    python scripts/prepare_robocasa_assets.py --fixtures cabinets,handle_bar ...

Then upload (needs write access to the org)::

    hf upload UniEnvOrg/unienv_genesis_collection_assets \\
        <output>/objects/robocasa objects/robocasa --repo-type dataset
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np

# Repository root (script lives in <repo>/scripts) — the sanity gate drives
# the collection's RoboCasaFixture descriptor.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

UPSTREAM_URL = "https://robocasa.ai"
DEFAULT_HF_REPO = "jianzhang96/robocasa-assets"
OFFICIAL_HF_REPO = "robocasa/robocasa-assets"
DOWNLOAD_KITCHEN_ASSETS_DOC = "python -m robocasa.scripts.download_kitchen_assets --type all"

OUTPUT_ASSETS_SUBDIR = "objects/robocasa"
SHELLS_SUBDIR = "scenes/rooms/robocasa_kitchen"
HF_UPLOAD_REPO = "UniEnvOrg/unienv_genesis_collection_assets"

# Style baked into the composed fixtures (fixed per run — see module docstring).
CABINET_TEXTURE = "textures/wood/wood_grain_1.png"
HANDLE_TEXTURE = "textures/metals/bright_metal.png"
# RoboCasa's cabinet construction constants (robocasa/models/fixtures/cabinets.py).
CABINET_THICKNESS = 0.03
CABINET_DOOR_GAP = 0.003

# Bbox cross-check tolerances (plan §1.3 packaging-time gate).
BBOX_TOL_ABS = 0.03  # metres
BBOX_TOL_REL = 0.12  # fraction of the per-axis extent


# ---------------------------------------------------------------------------
# Fixture registry (the parameterized --fixtures selection)
# ---------------------------------------------------------------------------
#
# kind: "composed" (ported from robocasa's ProcGen composition) or "static"
# (copied fixture directory).  Sizes follow RoboCasa's kitchen layouts
# (bottom-row cabinets [0.65, 0.6, 0.84], upper hinge cabinets depth 0.4,
# counter runs [_, 0.65, 0.92]).  `expect_joints` gates on a non-empty joint
# set; fixtures without joints are fixed furniture.
FIXTURE_REGISTRY: Dict[str, dict] = {
    # --- composed: cabinets / drawer / panel / handles ----------------------
    "cabinet_hinge": dict(
        kind="composed", template="cabinets/cabinet_hinge.xml",
        size=(0.90, 0.40, 0.92), panel_type="slab", handle_type="bar",
        texture=CABINET_TEXTURE, expect_joints=True, family="cabinet",
        annotations=["furniture", "kitchen cabinet", "wooden double-door kitchen cabinet"],
    ),
    "cabinet_single": dict(
        kind="composed", template="cabinets/cabinet_single.xml",
        size=(0.65, 0.60, 0.84), panel_type="slab", handle_type="bar",
        orientation="right", texture=CABINET_TEXTURE,
        expect_joints=True, family="cabinet",
        annotations=["furniture", "kitchen cabinet", "single-door kitchen cabinet"],
    ),
    "drawer": dict(
        kind="composed", template="cabinets/drawer.xml",
        size=(0.65, 0.60, 0.30), panel_type="slab", handle_type="bar",
        texture=CABINET_TEXTURE, expect_joints=True, family="drawer",
        annotations=["furniture", "kitchen drawer", "kitchen drawer unit with bar handle"],
    ),
    "panel": dict(
        kind="composed", template="cabinets/panel.xml",
        size=(0.40, 0.60, 0.84), panel_type="slab", handle_type=None,
        solid_body=True, texture=CABINET_TEXTURE,
        expect_joints=False, family="cabinet",
        annotations=["furniture", "kitchen cabinet", "solid kitchen cabinet filler panel"],
    ),
    "handle_bar": dict(
        kind="composed", template="handles/bar_handle.xml", texture=HANDLE_TEXTURE,
        expect_joints=False, family="handle",
        annotations=["furniture", "cabinet handle", "metal bar cabinet handle"],
    ),
    "handle_boxed": dict(
        kind="composed", template="handles/boxed_handle.xml", texture=HANDLE_TEXTURE,
        expect_joints=False, family="handle",
        annotations=["furniture", "cabinet handle", "metal boxed cabinet handle"],
    ),
    "handle_knob": dict(
        kind="composed", template="handles/knob_handle.xml", texture=HANDLE_TEXTURE,
        expect_joints=False, family="handle",
        annotations=["furniture", "cabinet handle", "round metal cabinet knob"],
    ),
    # --- static: appliances / sinks / counters ------------------------------
    "dishwasher_basic_stainless": dict(
        kind="static", source="dishwashers/basic_stainless",
        expect_joints=False, family="dishwasher",
        annotations=["furniture", "kitchen appliance", "stainless steel dishwasher"],
    ),
    "oven_bosch": dict(
        kind="static", source="ovens/bosch",
        expect_joints=False, family="oven",
        annotations=["furniture", "kitchen appliance", "built-in kitchen oven"],
    ),
    "fridge_four_doors": dict(
        kind="static", source="fridges/four_doors",
        expect_joints=False, family="fridge",
        annotations=["furniture", "kitchen appliance", "four-door refrigerator"],
    ),
    "fridge_full_height_doors": dict(
        kind="static", source="fridges/full_height_doors",
        expect_joints=False, family="fridge",
        annotations=["furniture", "kitchen appliance", "full-height-door refrigerator"],
    ),
    "fridge_modern": dict(
        kind="static", source="fridges/modern",
        expect_joints=False, family="fridge",
        annotations=["furniture", "kitchen appliance", "modern refrigerator"],
    ),
    "microwave_pack_1": dict(
        kind="static", source="microwaves/pack_1",
        expect_joints=True, family="microwave",
        annotations=["furniture", "kitchen appliance", "microwave with hinged door"],
    ),
    "microwave_standard": dict(
        kind="static", source="microwaves/standard",
        expect_joints=True, family="microwave",
        annotations=["furniture", "kitchen appliance", "microwave with side-hinged door"],
    ),
    "stove_simple_gas": dict(
        kind="static", source="stoves/simple_gas",
        expect_joints=False, family="stove",
        annotations=["furniture", "kitchen appliance", "gas stove with control knobs"],
    ),
    "stove_dual_gas": dict(
        kind="static", source="stoves/dual_gas",
        expect_joints=True, family="stove",
        annotations=["furniture", "kitchen appliance", "dual gas stove with control knobs"],
    ),
    "sink_white_sink": dict(
        kind="static", source="sinks/white_sink",
        expect_joints=True, family="sink",
        annotations=["furniture", "kitchen sink", "white kitchen sink with faucet handles"],
    ),
    "sink_2_bins_farmhouse": dict(
        kind="static", source="sinks/2_bins_farmhouse",
        expect_joints=True, family="sink",
        annotations=["furniture", "kitchen sink", "double-bin farmhouse kitchen sink"],
    ),
    "counter": dict(
        kind="static", source="counters/counter",
        expect_joints=False, family="counter",
        annotations=["furniture", "kitchen counter", "kitchen counter module with marble top"],
    ),
    "counter_with_opening": dict(
        kind="static", source="counters/counter_with_opening",
        expect_joints=False, family="counter",
        annotations=["furniture", "kitchen counter", "kitchen counter module with opening"],
    ),
}

FIXTURE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "cabinets": ("cabinet_hinge", "cabinet_single", "drawer", "panel"),
    "handles": ("handle_bar", "handle_boxed", "handle_knob"),
    "appliances": (
        "dishwasher_basic_stainless", "oven_bosch",
        "fridge_four_doors", "fridge_full_height_doors", "fridge_modern",
        "microwave_pack_1", "microwave_standard",
        "stove_simple_gas", "stove_dual_gas",
        "sink_white_sink", "sink_2_bins_farmhouse",
    ),
    "counters": ("counter", "counter_with_opening"),
}
ALL_FIXTURES: Tuple[str, ...] = (
    FIXTURE_GROUPS["cabinets"] + FIXTURE_GROUPS["handles"]
    + FIXTURE_GROUPS["appliances"] + FIXTURE_GROUPS["counters"]
)
# Plan risk #1: verify cabinet hinge + drawer FIRST (cheapest signal).
CHEAPEST_SIGNAL: Tuple[str, str] = ("cabinet_hinge", "drawer")


def resolve_fixture_selection(spec: Sequence[str]) -> List[str]:
    """Expand a ``--fixtures`` selection into fixture ids (registry order)."""
    selected: set = set()
    for token in spec:
        token = token.strip().lower()
        if not token:
            continue
        if token == "all":
            selected.update(ALL_FIXTURES)
        elif token in FIXTURE_GROUPS:
            selected.update(FIXTURE_GROUPS[token])
        elif token in FIXTURE_REGISTRY:
            selected.add(token)
        else:
            raise ValueError(
                f"Unknown fixture or group {token!r}. Known groups: "
                f"{sorted(FIXTURE_GROUPS)}; known ids: {list(ALL_FIXTURES)}"
            )
    return [f for f in ALL_FIXTURES if f in selected]


# ---------------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------------


def _extract_zip(zip_path: Path, dest: Path) -> None:
    print(f"  extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)


def _fetch_from_hub(repo_id: str, scratch: Path) -> Path:
    import huggingface_hub

    print(f"Downloading RoboCasa assets from Hugging Face repo {repo_id!r} ...")
    local = Path(
        huggingface_hub.snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=["fixtures.zip", "textures.zip"],
            local_dir=str(scratch / "hf_snapshot"),
        )
    )
    for zip_name in ("fixtures.zip", "textures.zip"):
        zip_path = local / zip_name
        if not zip_path.is_file():
            raise FileNotFoundError(
                f"{repo_id} does not provide {zip_name}. Alternative sources: a "
                f"local robocasa installation ({DOWNLOAD_KITCHEN_ASSETS_DOC}, "
                f"output under robocasa/models/assets) passed via --source."
            )
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(scratch)
    return scratch


def resolve_source(source: Optional[Path], repo_id: str, scratch: Path) -> Path:
    """Return the source root holding ``fixtures/`` and ``textures/``.

    ``source`` may be the robocasa ``models/assets`` download directory, or any
    directory holding the extracted trees or their ``fixtures.zip`` /
    ``textures.zip``.  When omitted, the assets are fetched from the Hub.
    """
    if source is None:
        return _fetch_from_hub(repo_id, scratch)

    source = source.expanduser().resolve()
    if (source / "fixtures" / "cabinets").is_dir():
        if not (source / "textures" / "wood").is_dir():
            raise FileNotFoundError(
                f"Source {source} has 'fixtures/' but no 'textures/' tree; the "
                "fixture MJCFs reference shared textures across the tree."
            )
        return source
    if (source / "fixtures.zip").is_file() and (source / "textures.zip").is_file():
        print(f"Extracting fixtures.zip / textures.zip from {source} ...")
        with zipfile.ZipFile(source / "fixtures.zip") as zf:
            zf.extractall(scratch)
        with zipfile.ZipFile(source / "textures.zip") as zf:
            zf.extractall(scratch)
        return scratch
    raise FileNotFoundError(
        f"{source} is not a RoboCasa assets root: expected 'fixtures/cabinets' "
        f"(or fixtures.zip + textures.zip). Alternative sources: a robocasa "
        f"installation ({DOWNLOAD_KITCHEN_ASSETS_DOC}, output under "
        f"robocasa/models/assets), or omit --source to fetch from the Hub."
    )


# ---------------------------------------------------------------------------
# MJCF utilities
# ---------------------------------------------------------------------------


def _find_by_name(root: ET.Element, tag: str, name: str) -> Optional[ET.Element]:
    """First element of *tag* under *root* whose ``name`` attribute is *name*."""
    for element in root.iter(tag):
        if element.get("name") == name:
            return element
    return None


def compile_check(model_xml: Path) -> List[str]:
    """Compile the packaged MJCF with MuJoCo, repairing degenerate mesh assets.

    A few upstream RoboCasa meshes are degenerate (zero volume — e.g.
    ``fridge_modern``'s ``model_1_vis``, ``sink_white_sink``'s
    ``model_0_vis``); MuJoCo rejects them with ``mesh volume is too small``.
    Since the fixtures are welded-root furniture whose dynamics don't depend
    on those visual meshes' inertia, the packaging applies MuJoCo's
    ``inertia="shell"`` on the offending mesh asset and recompiles.  Returns
    the list of patched mesh names.
    """
    import mujoco

    patched: List[str] = []
    for _ in range(3):
        try:
            mujoco.MjModel.from_xml_path(str(model_xml))
            return patched
        except ValueError as exc:
            match = re.search(r"mesh volume is too small: (\S+)", str(exc))
            if match is None:
                raise
            mesh_name = match.group(1).rstrip(".")
            root = ET.parse(model_xml).getroot()
            mesh = _find_by_name(root, "mesh", mesh_name)
            if mesh is None or mesh.get("inertia") is not None:
                raise
            mesh.set("inertia", "shell")
            ET.indent(root)
            model_xml.write_text(ET.tostring(root, encoding="unicode"))
            patched.append(mesh_name)
            print(f"    patched degenerate mesh {mesh_name!r} with inertia=\"shell\"")
    return patched


def _a2s(array: Sequence[float]) -> str:
    """Format a numeric sequence the way MJCF vector attributes expect."""
    return " ".join(f"{float(v):.10g}" for v in array)


def _s2a(text: str) -> np.ndarray:
    return np.asarray([float(v) for v in text.split()], dtype=float)


def strip_freejoint(root: ET.Element) -> List[str]:
    """Strip free joints from root bodies (fixed furniture — welded root).

    MJCF free joints are only meaningful at the root; a fixed fixture's root
    must be jointless so Genesis welds it to the world.  Free joints found on
    nested bodies raise (the fixture is dropped by the caller) — no RoboCasa
    fixture ships one.
    """
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF has no <worldbody>")
    removed: List[str] = []

    def _is_free(element: ET.Element) -> bool:
        return element.tag == "freejoint" or (
            element.tag == "joint" and element.get("type") == "free"
        )

    def _walk(body: ET.Element, is_root: bool) -> None:
        for joint in list(body):
            if _is_free(joint):
                if not is_root:
                    raise ValueError(
                        f"free joint {joint.get('name') or joint.tag!r} on a "
                        "nested body — unsupported for fixed furniture"
                    )
                removed.append(joint.get("name") or joint.tag)
                body.remove(joint)
        for child in body.findall("body"):
            _walk(child, False)

    for body in worldbody.findall("body"):
        _walk(body, True)
    return removed


def collision_geoms(root: ET.Element) -> List[ET.Element]:
    """Geoms that carry collision (nonzero contype/conaffinity — MuJoCo defaults)."""
    result = []
    for geom in root.iter("geom"):
        contype = int(geom.get("contype") or 0)
        conaffinity = int(geom.get("conaffinity") or 0)
        if contype or conaffinity or (
            "contype" not in geom.attrib and "conaffinity" not in geom.attrib
        ):
            result.append(geom)
    return result


# ---------------------------------------------------------------------------
# Composed fixtures: port of robocasa/models/fixtures/{cabinets,cabinet_panels,handles}.py
# ---------------------------------------------------------------------------


def _geom_element(
    name: str,
    geom_type: str,
    size: Sequence[float],
    pos: Sequence[float],
    euler: Optional[Sequence[float]] = None,
    quat: Optional[Sequence[float]] = None,
    material: Optional[str] = None,
    **extra,
) -> ET.Element:
    attributes: Dict[str, str] = dict(
        name=name, type=geom_type, size=_a2s(size), pos=_a2s(pos)
    )
    if material is not None:
        attributes["material"] = material
    if euler is not None:
        attributes["euler"] = _a2s(euler)
    if quat is not None:
        attributes["quat"] = _a2s(quat)
    attributes.update({key: str(value) for key, value in extra.items()})
    return ET.Element("geom", attributes)


def _panel_geometry(
    panel_type: str, size: Tuple[float, float, float],
) -> List[Tuple[str, Tuple[float, float, float], Tuple[float, float, float]]]:
    """(name, panel-frame half-sizes, pos) of a cabinet panel's geoms.

    Port of ``robocasa.models.fixtures.cabinet_panels`` (Slab/Shaker/Raised
    ``_create_panel``) with robocasa's defaults (trim_th=0.02, trim_size=0.08,
    raised_gap=0.01).  Sizes are in the panel frame [w, d, h]; the door/trim
    geoms carry robocasa's euler 90° x-rotation ("rotation trick" — see
    ``set_geom_dimensions(rotated=True)``), which swaps the depth/height axes
    at emission time.
    """
    w, d, h = [v / 2.0 for v in size]
    trim_th = CABINET_THICKNESS / 2.0
    trim_size = 0.08 / 2.0
    door_th = d - trim_th
    door_y = trim_th / 2.0
    trim_y = -door_th / 2.0

    def _rotated(geom_size: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Robocasa's ``set_geom_dimensions(rotated=True)`` texture trick.

        Door/trim geoms carry euler (90° about x); their authored panel-frame
        half-sizes [w/2, d/2, h/2] are re-ordered to [w/2, h/2, d/2] so the
        rotated geom's world footprint is [w, d, h].
        """
        return (geom_size[0], geom_size[2], geom_size[1])

    if panel_type == "slab":
        return [("door", _rotated((w, d, h)), (0.0, 0.0, 0.0))]
    if panel_type == "shaker":
        return [
            ("door", _rotated((w - 2 * trim_size, door_th, h - 2 * trim_size)), (0.0, door_y, 0.0)),
            ("trim_left", _rotated((trim_size, trim_th, h)), (-w + trim_size, trim_y, 0.0)),
            ("trim_right", _rotated((trim_size, trim_th, h)), (w - trim_size, trim_y, 0.0)),
            ("trim_top", _rotated((w - 2 * trim_size, trim_th, trim_size)), (0.0, trim_y, h - trim_size)),
            ("trim_bottom", _rotated((w - 2 * trim_size, trim_th, trim_size)), (0.0, trim_y, -h + trim_size)),
        ]
    if panel_type == "raised":
        raised_gap = 0.01 / 2.0
        return [
            ("door", _rotated((w - 2 * trim_size, door_th, h - 2 * trim_size)), (0.0, door_y, 0.0)),
            ("trim_left", _rotated((trim_size, trim_th, h)), (-w + trim_size, trim_y, 0.0)),
            ("trim_right", _rotated((trim_size, trim_th, h)), (w - trim_size, trim_y, 0.0)),
            ("trim_top", _rotated((w - 2 * trim_size, trim_th, trim_size)), (0.0, trim_y, h - trim_size)),
            ("trim_bottom", _rotated((w - 2 * trim_size, trim_th, trim_size)), (0.0, trim_y, -h + trim_size)),
            # Raised portion: inset from the trims, protruding forward.
            ("door_raised",
             _rotated((w - 2 * trim_size - 2 * raised_gap, trim_th,
                       h - 2 * trim_size - 2 * raised_gap)),
             (0.0, -(d - trim_th) / 2.0, 0.0)),
        ]
    raise NotImplementedError(f"Unsupported panel_type {panel_type!r} (slab/shaker/raised)")


def _handle_geometry(
    handle_type: Optional[str],
    panel_w: float,
    panel_h: float,
    orientation: str,
) -> Optional[Tuple[Tuple[float, float, float], List[Tuple[str, str, Tuple[float, ...], Tuple[float, float, float]]]]]:
    """Handle body position + geoms (port of ``robocasa.fixtures.handles``).

    Returns ``(body_pos, [(name, type, half_size, pos), ...])`` — sizes in the
    door/panel frame.  ``orientation`` is ``"vertical"`` for doors and
    ``"horizontal"`` for drawers (robocasa's ``Drawer`` sets it).
    """
    if handle_type is None:
        return None
    # robocasa handle padding (BarHandle/BoxedHandle default 0.04, Knob 0.07).
    handle_pad = 0.04 if handle_type in ("bar", "boxed") else 0.07
    length = 0.24
    if handle_type in ("bar", "boxed") and panel_h < length + 2 * handle_pad:
        length = panel_h - 2 * handle_pad
        if length <= 0:
            raise ValueError(
                f"Panel height {panel_h} is too small for {handle_type} handles "
                f"with padding {handle_pad}"
            )

    if handle_type == "bar":
        offset = length / 2.0 * 0.60  # robocasa BarHandle._create_handle
        conn_len = 0.05
        geoms: List[Tuple[str, str, Tuple[float, ...], Tuple[float, float, float]]] = [
            ("handle", "cylinder", (0.013, length / 2.0), (0.0, -conn_len, 0.0)),
            ("handle_connector_top", "cylinder", (0.008, conn_len / 2.0),
             (0.0, -conn_len / 2.0, offset)),
            ("handle_connector_bottom", "cylinder", (0.008, conn_len / 2.0),
             (0.0, -conn_len / 2.0, -offset)),
        ]
    elif handle_type == "boxed":
        conn_len = 0.05
        connector_depth = (conn_len / 2.0) - 0.01
        connector_zpos = (length / 2.0) - 0.01
        geoms = [
            ("handle", "box", (0.01, 0.01, length / 2.0), (0.0, -conn_len, 0.0)),
            ("handle_connector_top", "box", (0.01, 0.01, connector_depth),
             (0.0, -conn_len / 2.0, connector_zpos)),
            ("handle_connector_bottom", "box", (0.01, 0.01, connector_depth),
             (0.0, -conn_len / 2.0, -connector_zpos)),
        ]
    elif handle_type == "knob":
        geoms = [("handle", "cylinder", (0.015, 0.017), (0.0, -0.017, 0.0))]
    else:
        raise NotImplementedError(f"Unsupported handle_type {handle_type!r}")

    if orientation == "horizontal":
        # robocasa: swap the connectors' x/z, lay the main handle flat.
        geoms = [
            (
                name,
                gtype,
                (s[0], s[2], s[1]) if gtype == "box" else s,
                (p[0], p[2], p[1]) if "connector" in name else p,
            )
            for name, gtype, s, p in geoms
        ]
    # robocasa CabinetPanel._add_handle placement (handle_hpercent=0.825,
    # handle_vpercent=0.80; drawers use handle_hpos/vpos="center").
    return (0.0, 0.0, 0.0), geoms


def _attach_handle(
    parent_body: ET.Element,
    handle_id: str,
    panel_w: float,
    panel_h: float,
    handle_type: Optional[str],
    orientation: str,
    handle_hpos: str,
    handle_vpos: str,
) -> None:
    """Append a composed handle body to *parent_body* (``CabinetPanel._add_handle``)."""
    if handle_type is None:
        return
    _, geoms = _handle_geometry(handle_type, panel_w, panel_h, orientation)
    hpercent, vpercent = 0.825, 0.80
    if handle_hpos == "left":
        hx = -(panel_w / 2.0) * hpercent
    elif handle_hpos == "right":
        hx = (panel_w / 2.0) * hpercent
    elif handle_hpos == "center":
        hx = 0.0
    else:
        raise NotImplementedError(f"Unsupported handle_hpos {handle_hpos!r}")
    if handle_vpos == "bottom":
        handle_height = 0.24 if orientation == "vertical" else 0.05
        hz = (-(panel_h / 2.0) + handle_height / 2.0) * vpercent
    elif handle_vpos == "center":
        hz = 0.0
    else:
        raise NotImplementedError(f"Unsupported handle_vpos {handle_vpos!r}")

    body = ET.SubElement(parent_body, "body", {"name": handle_id, "pos": _a2s((hx, 0.0, hz))})
    for name, gtype, size, pos in geoms:
        body.append(_geom_element(
            f"{handle_id}_{name}", gtype, size, pos,
            euler=(0.0, 1.5708, 0.0) if (name == "handle" and orientation == "horizontal") else None,
        ))


def _add_panel(
    parent_body: ET.Element,
    panel_id: str,
    panel_type: str,
    panel_size: Tuple[float, float, float],
    pos: Sequence[float],
    handle_type: Optional[str],
    handle_hpos: str,
    handle_vpos: str,
    handle_orientation: str = "vertical",
) -> None:
    """Robocasa ``Cabinet._add_door``: a panel body (door geoms + handle)."""
    panel_size = (
        panel_size[0] - CABINET_DOOR_GAP,
        panel_size[1],
        panel_size[2] - CABINET_DOOR_GAP,
    )
    body = ET.SubElement(
        parent_body, "body", {"name": panel_id, "pos": _a2s(pos)}
    )
    door_euler = (1.5708, 0.0, 0.0)
    for name, size, geom_pos in _panel_geometry(panel_type, panel_size):
        body.append(_geom_element(
            f"{panel_id}_{name}", "box", size, geom_pos, euler=door_euler, material="mat",
        ))
    _attach_handle(
        body, f"{panel_id}_handle", panel_size[0], panel_size[2],
        handle_type, handle_orientation, handle_hpos, handle_vpos,
    )


def compose_fixture(fixture_id: str, spec: dict, source: Path) -> ET.Element:
    """Build the composed fixture's MJCF tree (robocasa ProcGen port).

    Ports ``SingleCabinet`` / ``HingeCabinet`` / ``Drawer`` / ``PanelCabinet``
    ``_create_cab`` (``robocasa/models/fixtures/cabinets.py``) for the fixed
    style in *spec*: housing geoms resized to ``size``, door panel(s) added to
    the jointed body/bodies, handles composed per ``CabinetPanel._add_handle``.
    Returns the ``<mujoco>`` element; the texture asset points at the
    source-root-relative style texture (copied directory-local at packaging).
    """
    template_path = source / "fixtures" / spec["template"]
    root = ET.parse(template_path).getroot()
    _ensure_radian_compiler(root)
    _set_texture(root, spec["texture"])

    handle_type = spec.get("handle_type", "bar")
    th = CABINET_THICKNESS / 2.0

    if spec["template"] == "cabinets/cabinet_hinge.xml":
        # HingeCabinet._create_cab.
        x, y, z = [dim / 2.0 for dim in spec["size"]]
        body_left = _find_by_name(root, "body", "hingeleftdoor")
        body_right = _find_by_name(root, "body", "hingerightdoor")
        body_left.set("pos", _a2s((0.0, 0.0, 0.0)))
        body_right.set("pos", _a2s((0.0, 0.0, 0.0)))
        joint_left = _find_by_name(root, "joint", "leftdoorhinge")
        joint_right = _find_by_name(root, "joint", "rightdoorhinge")
        joint_left.set("pos", _a2s((-x + th, -y, 0.0)))
        joint_right.set("pos", _a2s((x - th, -y, 0.0)))

        sizes = {
            "top": (x, y - th, th), "bottom": (x, y - th, th),
            "back": (x - 2 * th, th, z - 2 * th),
            "left": (th, y - th, z - 2 * th), "right": (th, y - th, z - 2 * th),
        }
        positions = {
            "top": (0.0, th, z - th), "bottom": (0.0, th, -z + th),
            "back": (0.0, y - th, 0.0),
            "left": (-x + th, th, 0.0), "right": (x - th, th, 0.0),
        }
        housing = {name: _find_by_name(root, "geom", name) for name in sizes}
        _apply_dimensions(sizes, positions, housing, rotated=True)

        door_x = {"left": -x / 2.0, "right": x / 2.0}
        for side in ("left", "right"):
            _add_panel(
                body_left if side == "left" else body_right,
                f"{fixture_id}_{side}_door", spec["panel_type"],
                panel_size=(x, th * 2.0, z * 2.0),
                pos=(door_x[side], -y + th, 0.0),
                handle_type=handle_type,
                handle_hpos="right" if side == "left" else "left",
                handle_vpos="bottom",
            )
    elif spec["template"] == "cabinets/cabinet_single.xml":
        # SingleCabinet._create_cab.
        x, y, z = [dim / 2.0 for dim in spec["size"]]
        sizes = {
            "top": (x, y - th, th), "bottom": (x, y - th, th),
            "back": (x - 2 * th, th, z - 2 * th),
            "left": (th, y - th, z - 2 * th), "right": (th, y - th, z - 2 * th),
        }
        positions = {
            "top": (0.0, th, z - th), "bottom": (0.0, th, -z + th),
            "back": (0.0, y - th, 0.0),
            "left": (-x + th, th, 0.0), "right": (x - th, th, 0.0),
        }
        housing = {name: _find_by_name(root, "geom", name) for name in sizes}
        _apply_dimensions(sizes, positions, housing, rotated=True)

        body = _find_by_name(root, "body", "hingedoor")
        body.set("pos", _a2s((0.0, 0.0, 0.0)))
        joint = _find_by_name(root, "joint", "doorhinge")
        if spec.get("orientation", "right") == "right":
            joint.set("pos", _a2s((x - th, -y, 0.0)))
            joint.set("range", _a2s((0.0, 1.57)))
        else:
            joint.set("pos", _a2s((-x + th, -y, 0.0)))
            joint.set("range", _a2s((-1.57, 0.0)))
        _add_panel(
            body, f"{fixture_id}_door", spec["panel_type"],
            panel_size=(x * 2.0, th * 2.0, z * 2.0),
            pos=(0.0, -y + th, 0.0),
            handle_type=handle_type,
            handle_hpos="right" if spec.get("orientation", "right") == "left" else "left",
            handle_vpos="bottom",
        )
    elif spec["template"] == "cabinets/drawer.xml":
        # Drawer._create_cab.
        x, y, z = [dim / 2.0 for dim in spec["size"]]
        ix = x - 2 * th - 0.001  # inner box half-width
        iy = y - 2 * th
        iz = z - 2 * th - 0.001  # inner box half-height
        sizes = {
            "top": (x, y - th, th), "bottom": (x, y - th, th),
            "back": (x - 2 * th, th, z - 2 * th),
            "left": (th, y - th, z - 2 * th), "right": (th, y - th, z - 2 * th),
            "inner_bottom": (ix, iy, th), "inner_back": (ix - 2 * th, th, iz - 2 * th),
            "inner_left": (th, iy, iz - 2 * th), "inner_right": (th, iy, iz - 2 * th),
        }
        positions = {
            "top": (0.0, th, z - th), "bottom": (0.0, th, -z + th),
            "back": (0.0, y - th, 0.0),
            "left": (-x + th, th, 0.0), "right": (x - th, th, 0.0),
            "inner_bottom": (0.0, 0.0, -iz + th), "inner_back": (0.0, iy - th, 0.0),
            "inner_left": (-ix + th, 0.0, 0.0), "inner_right": (ix - th, 0.0, 0.0),
        }
        housing = {name: _find_by_name(root, "geom", name) for name in sizes}
        _apply_dimensions(sizes, positions, housing, rotated=True)

        body = _find_by_name(root, "body", "inner_box")
        body.set("pos", _a2s((0.0, 0.0, 0.0)))
        joint = _find_by_name(root, "joint", "slidejoint")
        joint.set("pos", _a2s((0.0, -y, 0.0)))
        joint.set("range", _a2s((-y * 2.0, 0.0)))
        _add_panel(
            body, f"{fixture_id}_door", spec["panel_type"],
            panel_size=(x * 2.0, th * 2.0, z * 2.0),
            pos=(0.0, -y + th, 0.0),
            handle_type=handle_type,
            handle_hpos="center", handle_vpos="center",
            handle_orientation="horizontal",
        )
    elif spec["template"] == "cabinets/panel.xml":
        # PanelCabinet._create_cab: optional solid body + front panel, no handle.
        x, y, z = [dim / 2.0 for dim in spec["size"]]
        object_body = _find_by_name(root, "body", "object")
        if spec.get("solid_body"):
            object_body.append(_geom_element(
                f"{fixture_id}_body", "box", (x, y - th, z), (0.0, th, 0.0),
                density="10", rgba="0.5 0 0 1",
            ))
        _add_panel(
            object_body, f"{fixture_id}_door", spec["panel_type"],
            panel_size=(x * 2.0, th * 2.0, z * 2.0),
            pos=(0.0, -y + th, 0.0),
            handle_type=None, handle_hpos="center", handle_vpos="center",
        )
    elif spec["template"].startswith("handles/"):
        # Handle fixtures ship their authored geometry; only the texture is
        # normalized to the registry's metal texture (robocasa Handle default).
        root.set("model", f"handle_{spec['template'].split('/')[-1].removesuffix('.xml')}")
    else:  # pragma: no cover — registry/template mismatch
        raise ValueError(f"Unknown composed template {spec['template']!r}")

    return root


def _ensure_radian_compiler(root: ET.Element) -> None:
    """Ensure ``<compiler angle="radian"/>`` on the composed MJCF.

    RoboCasa's fixture templates write euler/range attribute values in
    *radians* (e.g. the door panels' ``euler="1.5708 0 0"``); most templates
    declare ``<compiler angle="radian"/>`` but a few (``cabinet_hinge.xml``)
    omit it — upstream's robosuite loader normalises the angle unit when it
    parses fixtures.  A self-contained packaged MJCF must declare it so
    MuJoCo/Genesis read the authored values as radians (degrees would turn a
    90° door into 1.57°).
    """
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler", {"angle": "radian"})
        root.insert(0, compiler)
    compiler.set("angle", "radian")


def _set_texture(root: ET.Element, file_ref: str) -> None:
    """Point the template's ``tex`` texture at *file_ref*.

    Port of robocasa's ``Cabinet._set_texture`` / ``Handle._set_texture``
    (minus the runtime name disambiguation: a self-contained fixture needs no
    cross-asset texture namespacing).
    """
    texture = _find_by_name(root, "texture", "tex")
    if texture is None:
        raise ValueError("Fixture template has no <texture name='tex'>")
    texture.set("file", file_ref)
    for material in root.iter("material"):
        if material.get("texture") is not None:
            material.set("texture", "tex")


def copy_asset_refs(
    root: ET.Element, source: Path, template_dir: Path, fixture_dir: Path,
) -> List[str]:
    """Copy a composed fixture's referenced assets into its fixture directory.

    Composed fixtures reference their style textures by source-root-relative
    paths (``textures/wood/wood_grain_1.png``) or template-relative paths
    (``../../textures/ceramic.png``).  Each reference is resolved (template
    directory first, then the source root — the two bases RoboCasa's loader
    uses), copied under the *same relative subpath* inside the fixture
    directory and left as a directory-local ref.
    """
    source = source.resolve()
    template_dir = template_dir.resolve()
    copied: List[str] = []
    for element in root.iter():
        ref = element.get("file")
        if ref is None or element.tag == "include":
            continue
        candidate = Path(ref)
        source_file = None
        for base in (template_dir, source):
            resolved = candidate if candidate.is_absolute() else base / candidate
            if resolved.is_file():
                source_file = resolved
                break
        if source_file is None:
            raise FileNotFoundError(
                f"{fixture_dir.name}: <{element.tag}> references missing file {ref!r}"
            )
        dest = fixture_dir / ref
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, dest)
        copied.append(ref)
    return copied


def _apply_dimensions(
    sizes: Dict[str, Sequence[float]],
    positions: Dict[str, Sequence[float]],
    geoms: Dict[str, Optional[ET.Element]],
    rotated: bool = False,
) -> None:
    """Robocasa's ``set_geom_dimensions`` (``robocasa/utils/object_utils.py``).

    ``rotated=True`` applies robocasa's texture-uniformity trick for
    ``door``/``trim`` geoms: their authored euler (90° about x) swaps the
    depth/height axes, so panel-frame half-sizes are re-ordered to [x, z, y].
    """
    if rotated:
        sizes = {
            key: ([s[0], s[2], s[1]] if ("door" in key or "trim" in key) else list(s))
            for key, s in sizes.items()
        }
    for name, pos in positions.items():
        geom = geoms[name]
        if geom is None:
            raise ValueError(f"Did not find geom: {name}")
        geom.set("pos", _a2s(pos))
        geom.set("size", _a2s(sizes[name]))


# ---------------------------------------------------------------------------
# Primitive baking (composed fixtures: box/cylinder geoms -> mesh geoms)
# ---------------------------------------------------------------------------


def _face_uvs(n_faces: int) -> np.ndarray:
    return np.tile(
        np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32), (n_faces, 1)
    )


def _write_box_obj(path: Path, half_extents: Sequence[float]) -> None:
    """Axis-aligned box OBJ (half-extents), per-vertex UVs, no material.

    UVs are per-*vertex* (corner-derived), so all six faces share the same
    eight vertices: the mesh stays watertight (clean MuJoCo/trimesh collision
    and inertia) while still carrying texture coordinates for the material.
    """
    hx, hy, hz = (float(v) for v in half_extents)
    vertices = [
        (-hx, -hy, -hz), (+hx, -hy, -hz), (+hx, +hy, -hz), (-hx, +hy, -hz),
        (-hx, -hy, +hz), (+hx, -hy, +hz), (+hx, +hy, +hz), (-hx, +hy, +hz),
    ]
    # Per-vertex UV corners; shared vertices get one UV, keeping the mesh
    # watertight (texture continuity across faces rather than per-face tiles).
    uvs = [(v[0] > 0.0, v[2] > 0.0) for v in vertices]
    # Triangles (1-based vertex indices), outward windings per face.
    faces = [
        (1, 4, 3), (1, 3, 2),  # -z
        (2, 3, 7), (2, 7, 6),  # +x
        (3, 4, 8), (3, 8, 7),  # +y
        (1, 2, 6), (1, 6, 5),  # -y
        (4, 1, 5), (4, 5, 8),  # -x
        (5, 6, 7), (5, 7, 8),  # +z
    ]
    lines = [f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}" for v in vertices]
    lines += [f"vt {float(u):.9g} {float(v):.9g}" for u, v in uvs]
    lines += [f"f {f[0]}/{f[0]} {f[1]}/{f[1]} {f[2]}/{f[2]}" for f in faces]
    path.write_text("\n".join(lines) + "\n")


def _write_cylinder_obj(path: Path, radius: float, half_height: float) -> None:
    import trimesh

    mesh = trimesh.creation.cylinder(radius=radius, height=2.0 * half_height, sections=24)
    mesh.visual = trimesh.visual.TextureVisuals(uv=_face_uvs(len(mesh.faces)))
    text = trimesh.exchange.obj.export_obj(mesh)
    # Strip material bookkeeping — the MJCF material drives the appearance.
    lines = [
        line for line in text.splitlines()
        if not line.startswith(("mtllib", "usemtl", "o ", "g "))
    ]
    path.write_text("\n".join(lines) + "\n")


def bake_primitive_geoms(root: ET.Element, fixture_dir: Path) -> List[str]:
    """Bake box/cylinder geoms into exact-dimension mesh geoms (composed fixtures).

    The offline FK walker (``mjcf_link_meshes_at_config``, plan §1.3) reads
    ``<mesh>`` geoms only, so the composed fixtures' MuJoCo primitives become
    per-geom OBJ meshes with identical dimensions; collision turns into a
    convex mesh of the same silhouette.  Reset-region markers (transparent
    ``region``-class boxes) stay primitives.
    """
    meshes_dir = fixture_dir / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)
    assets = root.find("asset")
    if assets is None:
        assets = ET.SubElement(root, "asset")
    baked: List[str] = []
    for geom in root.iter("geom"):
        geom_type = geom.get("type")
        if geom_type not in ("box", "cylinder") or geom.get("size") is None:
            continue
        if geom.get("class") == "region" or "reg_" in (geom.get("name") or ""):
            continue
        size = _s2a(geom.get("size"))
        if (geom_type == "cylinder" and len(size) != 2) or (geom_type == "box" and len(size) != 3):
            continue
        geom_name = geom.get("name") or f"geom_{len(baked)}"
        mesh_name = f"{geom_name}_baked"
        file_name = f"meshes/{mesh_name}.obj"
        obj_path = fixture_dir / file_name
        if geom_type == "box":
            _write_box_obj(obj_path, size)
        else:
            _write_cylinder_obj(obj_path, float(size[0]), float(size[1]))
        ET.SubElement(assets, "mesh", {
            "name": mesh_name, "file": file_name, "scale": "1 1 1",
        })
        geom.set("type", "mesh")
        geom.set("mesh", mesh_name)
        del geom.attrib["size"]
        baked.append(file_name)
    return baked


# ---------------------------------------------------------------------------
# Repackaging: copy referenced files + rewrite refs (incl. recursive includes)
# ---------------------------------------------------------------------------


class Repackager:
    """Copies an MJCF and everything it references into a fixture directory.

    Mirrors the reference resolution of ``genesis.utils.mjcf.build_model`` —
    the loader that will consume these files, and the semantics encoded in the
    collection's ``HubAssetArticulatedObject._mjcf_required_files``:

    * ``<include>`` files resolve relative to the *main* file's directory
      (recursively; a nested include accumulates on its parent include's
      relative path, as ``build_model`` does);
    * ``<mesh file=…>`` inside an included file is relative to the include's
      directory; other ``file=`` refs (textures, hfields, …) stay relative to
      the main file's directory;
    * directories come from the *main* file's ``<compiler>``
      (``assetdir``/``meshdir``/``texturedir``), absolutized against it.
    """

    def __init__(self, source_root: Path, out_root: Path):
        self.source_root = Path(source_root).resolve()
        self.out_root = Path(out_root).resolve()
        self.copied: Dict[str, str] = {}  # source abs path -> dest rel (posix)

    # -- path helpers --------------------------------------------------------
    def _dest_rel(self, source_file: Path, rel_base: Path) -> str:
        """Fixture-directory-local path for *source_file*.

        Files under the referencing fixture's own source directory keep their
        relative layout (``visuals/model_0.obj``); shared registries outside it
        (RoboCasa's cross-tree ``textures/…``) are copied under the same
        source-root-relative subpath, keeping every reference unique and
        directory-local.
        """
        source_file = source_file.resolve()
        rel_base = rel_base.resolve()
        if rel_base == source_file or rel_base in source_file.parents:
            return source_file.relative_to(rel_base).as_posix()
        if self.source_root in source_file.parents:
            return source_file.relative_to(self.source_root).as_posix()
        raise ValueError(
            f"Referenced file {source_file} lies outside the fixture directory "
            f"and the source root {self.source_root}."
        )

    def _copy(self, source_file: Path, out_dir: Path, rel_base: Path) -> str:
        source_file = source_file.resolve()
        key = str(source_file)
        if key in self.copied:
            return self.copied[key]
        dest_rel = self._dest_rel(source_file, rel_base)
        # dest_rel is relative to the *fixture* directory (what the rewritten
        # MJCF spells); the file lands inside the fixture directory itself so
        # each fixture directory stays self-contained.
        dest = out_dir / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, dest)
        self.copied[key] = dest_rel
        return dest_rel

    # -- reference rewriting --------------------------------------------------
    def _rewrite_file(
        self,
        xml_file: Path,
        out_dir: Path,
        include_parent: Optional[Path],
        main_dir: Optional[Path] = None,
    ) -> str:
        """Copy *xml_file*'s referenced files; return its rewritten MJCF text.

        ``include_parent`` is *xml_file*'s own include-relative path (relative
        to the fixture directory) when it was pulled in through
        ``<include>`` — nested references resolve off it like ``build_model``.
        ``main_dir`` is the main file's directory (the resolution base of
        non-mesh ``file=`` refs inside included files); defaults to
        *xml_file*'s directory for the top-level file.
        """
        root = ET.parse(xml_file).getroot()
        compiler = root.find("compiler")
        if main_dir is None:
            main_dir = xml_file.parent

        def _base_dir(attr: Optional[str], directory: Path) -> Path:
            if attr:
                resolved = Path(attr)
                return resolved if resolved.is_absolute() else directory / resolved
            return directory

        asset_dir = _base_dir(
            compiler.get("assetdir") if compiler is not None else None, xml_file.parent
        )
        mesh_dir = _base_dir(
            compiler.get("meshdir") if compiler is not None else None, xml_file.parent
        )
        texture_dir = _base_dir(
            compiler.get("texturedir") if compiler is not None else None, xml_file.parent
        )

        for element in root.iter():
            ref = element.get("file")
            if ref is None or element.tag == "include":
                continue
            resolved = ref
            if element.tag == "mesh" and include_parent is not None:
                resolved = str(include_parent.parent / ref)
            base = {"mesh": mesh_dir, "texture": texture_dir}.get(element.tag, asset_dir)
            source_file = Path(resolved)
            if not source_file.is_absolute():
                source_file = base / source_file
            if not source_file.is_file():
                raise FileNotFoundError(
                    f"{xml_file}: <{element.tag}> references missing file "
                    f"{ref!r} (resolved {source_file})"
                )
            # Files inside an included file's own directory keep that layout;
            # the rest are laid out relative to the main file's directory —
            # matching how the loader will resolve them again at runtime.
            rel_base = (
                xml_file.parent
                if (element.tag == "mesh" and include_parent is not None)
                else main_dir
            )
            element.set("file", self._copy(source_file, out_dir, rel_base))

        # The packaged MJCF is directory-local: every reference resolves
        # against the fixture directory, so the *main* file's compiler
        # assetdir/meshdir/texturedir (which the loader absolutizes against
        # the fixture directory) must not redirect them — drop those
        # attributes and keep the rest of the compiler element verbatim
        # (e.g. ``angle``).
        if compiler is not None:
            for attr in ("assetdir", "meshdir", "texturedir"):
                compiler.attrib.pop(attr, None)

        for include_element in root.findall("include"):
            include_ref = include_element.get("file")
            if include_ref is None:
                continue
            source_include = xml_file.parent / include_ref
            rel = Path(include_ref) if include_parent is None else include_parent / include_ref
            if not source_include.is_file():
                raise FileNotFoundError(
                    f"{xml_file}: <include> references missing file {include_ref!r}"
                )
            # The include file itself is copied (with its own references
            # rewritten, relative to its new directory — the loader resolves
            # included files' refs against the include's directory, mirroring
            # ``build_model`` / ``_mjcf_required_files``) and the include ref
            # becomes its directory-local name.
            dest_rel = self._dest_rel(source_include, xml_file.parent)
            include_element.set("file", dest_rel)
            self._rewrite_included_file(
                source_include, out_dir / dest_rel, rel, main_dir=main_dir,
            )
        ET.indent(root)
        return ET.tostring(root, encoding="unicode")

    def _rewrite_included_file(
        self,
        xml_file: Path,
        dest_file: Path,
        include_parent: Path,
        main_dir: Optional[Path] = None,
    ) -> None:
        """Copy an ``<include>`` file and rewrite its references in place.

        The include file is written to *dest_file* (inside the fixture
        directory); its ``<mesh file=…>`` references resolve relative to the
        include's own directory (the loader rewrites included ``<mesh>``
        elements against the include's path), other ``file=`` references
        against the main file's directory — as ``build_model`` does.
        """
        root = ET.parse(xml_file).getroot()
        compiler = root.find("compiler")
        if main_dir is None:
            main_dir = xml_file.parent

        def _base_dir(attr: Optional[str], directory: Path) -> Path:
            if attr:
                resolved = Path(attr)
                return resolved if resolved.is_absolute() else directory / resolved
            return directory

        asset_dir = _base_dir(
            compiler.get("assetdir") if compiler is not None else None, xml_file.parent
        )
        mesh_dir = _base_dir(
            compiler.get("meshdir") if compiler is not None else None, xml_file.parent
        )
        texture_dir = _base_dir(
            compiler.get("texturedir") if compiler is not None else None, xml_file.parent
        )

        for element in root.iter():
            ref = element.get("file")
            if ref is None or element.tag == "include":
                continue
            resolved = ref
            if element.tag == "mesh" and include_parent is not None:
                resolved = str(include_parent.parent / ref)
            base = {"mesh": mesh_dir, "texture": texture_dir}.get(element.tag, asset_dir)
            source_file = Path(resolved)
            if not source_file.is_absolute():
                source_file = base / source_file
            if not source_file.is_file():
                raise FileNotFoundError(
                    f"{xml_file}: <{element.tag}> references missing file "
                    f"{ref!r} (resolved {source_file})"
                )
            rel_base = (
                xml_file.parent
                if (element.tag == "mesh" and include_parent is not None)
                else main_dir
            )
            element.set("file", self._copy(source_file, dest_file.parent, rel_base))

        for include_element in root.findall("include"):
            include_ref = include_element.get("file")
            if include_ref is None:
                continue
            source_include = xml_file.parent / include_ref
            rel = include_parent / include_ref
            if not source_include.is_file():
                raise FileNotFoundError(
                    f"{xml_file}: <include> references missing file {include_ref!r}"
                )
            # Nested includes accumulate on the parent include's relative
            # path (the loader's convention — see ``_mjcf_required_files``).
            dest_rel = self._dest_rel(source_include, xml_file.parent)
            include_element.set("file", dest_rel)
            self._rewrite_included_file(
                source_include,
                dest_file.parent / dest_rel,
                rel,
                main_dir=main_dir,
            )
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        ET.indent(root)
        dest_file.write_text(ET.tostring(root, encoding="unicode"))

    def repackage(self, source_mjcf: Path, fixture_id: str) -> Path:
        """Package *source_mjcf* into ``out_root/<fixture_id>/model.xml``.

        All referenced meshes/textures (and the ``<include>`` chain, resolved
        recursively) are copied into the fixture directory and every ``file=``
        reference is rewritten directory-local.
        """
        out_dir = self.out_root / fixture_id
        out_dir.mkdir(parents=True, exist_ok=True)
        text = self._rewrite_file(source_mjcf, out_dir, None)
        (out_dir / "model.xml").write_text(text)
        return out_dir


def ensure_collision_geoms(
    root: ET.Element, fixture_dir: Path, fixture_id: str,
) -> Dict[str, object]:
    """Audit collision geoms; fall back to CoACD when a fixture ships none.

    RoboCasa fixtures ship authored collision geoms — used as-is (plan §2.1).
    A fixture whose every geom is visual-only (``contype == conaffinity == 0``)
    gets a convex decomposition of each link's offline-FK visual mesh via
    ``coacd`` (installed), emitted as ``collision_*.stl`` plus
    ``<geom type="mesh" mesh=… group="3">`` refs — mirroring the YCB pipeline.
    """
    geoms = collision_geoms(root)
    info: Dict[str, object] = {
        "n_collision_geoms": len(geoms),
        "collision_source": "authored",
    }
    if geoms:
        return info

    import coacd
    import trimesh

    from unienv_genesis_collection.objects.articulated_object import (
        mjcf_link_meshes_at_config,
    )

    link_meshes = mjcf_link_meshes_at_config(str(fixture_dir / "model.xml"))
    assets = root.find("asset")
    if assets is None:
        assets = ET.SubElement(root, "asset")
    index = 0
    for link_name in sorted(link_meshes):
        pieces = coacd.run_coacd(link_meshes[link_name])
        for vertices, faces in pieces:
            piece_mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces))
            file_name = f"collision_{index}.stl"
            piece_mesh.export(fixture_dir / file_name)
            ET.SubElement(assets, "mesh", {"name": f"collision_{index}", "file": file_name})
            parent = _find_by_name(root, "body", link_name)
            if parent is None:
                continue
            parent.append(ET.Element("geom", dict(
                name=f"{fixture_id}_collision_{index}", type="mesh",
                mesh=f"collision_{index}", group="3",
            )))
            index += 1
    info["collision_source"] = "coacd"
    info["n_collision_geoms"] = index
    return info


# ---------------------------------------------------------------------------
# Genesis sanity gate (plan risk #1)
# ---------------------------------------------------------------------------


def quat_to_matrix(quat: Sequence[float]) -> np.ndarray:
    """(w, x, y, z) quaternion → 3×3 rotation matrix."""
    w, x, y, z = (float(v) for v in quat)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class _WorldStub:
    """Minimal duck-typed ``GenesisWorld`` stand-in for ``load_into_scene``."""

    def __init__(self, scene) -> None:
        self.scene = scene


def _render_thumbnail(
    descriptor: RoboCasaFixture,
    fixture_id: str,
    render_dir: Path,
) -> str:
    """Render one standing thumbnail of *fixture_id* into *render_dir*.

    Near-copy of the ReplicaCAD gate's render block, but with the identity
    orientation: RoboCasa fixtures are authored z-up / floor-sitting (the
    catalog's ``standing_quaternion`` is None for every fixture), so the
    descriptor's local bbox doubles as the world bbox at the standing pose.
    Builds a SECOND throwaway scene — never the gate scene — places the
    fixture on a ground plane with its lowest corner at z = 0 and frames it
    from a shallow 3/4 offset camera for the texture-fidelity eyeball.

    Genesis must be initialised with the CUDA backend before calling (the
    Rasterizer camera does not work on CPU); never raises past the gate's
    per-fixture guard.
    """
    import genesis as gs

    local = descriptor.get_local_bounding_box()
    low = local.low.detach().cpu().numpy().astype(float)
    high = local.high.detach().cpu().numpy().astype(float)

    render_dir.mkdir(parents=True, exist_ok=True)
    render_scene = gs.Scene(show_viewer=False, renderer=gs.options.renderers.Rasterizer())
    try:
        z_offset = float(-low[2])  # lowest local corner sits on the plane
        center = ((low + high) / 2.0).copy()
        center[2] += z_offset  # the camera frames the *placed* fixture
        radius = max(max(hi - lo, 1e-3) for lo, hi in zip(low, high)) / 2.0

        render_scene.add_entity(gs.morphs.Plane())
        descriptor.load_into_scene(
            _WorldStub(render_scene),
            pos=(0.0, 0.0, z_offset),
            quat=(1.0, 0.0, 0.0, 0.0),  # identity — authored z-up orientation
        )
        # Shallow 3/4 view (mostly horizontal) so tall fixtures stay in frame.
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
        if hasattr(rgb, "detach"):
            rgb = rgb.detach().cpu().numpy()
        out_file = render_dir / f"{fixture_id}_standing.png"
        import imageio

        imageio.imwrite(out_file, np.asarray(rgb)[..., :3])
        return str(out_file)
    finally:
        try:
            render_scene.destroy()
        except Exception:  # noqa: BLE001 — teardown is best-effort
            pass


def sanity_gate(
    output_dir: Path,
    fixture_ids: Sequence[str],
    bbox_tol_abs: float = BBOX_TOL_ABS,
    bbox_tol_rel: float = BBOX_TOL_REL,
    render_dir: Optional[Path] = None,
) -> Tuple[List[dict], List[dict]]:
    """Load each packaged fixture in a throwaway Genesis scene (GPU).

    Per fixture: build a ``GenesisWorld`` scene, load through
    ``RoboCasaFixture.load_into_scene`` (the descriptor the room builder
    drives), assert a non-empty joint set where the registry expects
    articulation, and cross-check the offline MJCF-FK union bbox against
    ``entity.get_vAABB()`` at q = 0 (offline bbox transformed to the world
    frame by the entity's pose, tolerance ``max(abs_tol, rel_tol · extent)``
    per axis).  Surviving fixtures additionally get a standing thumbnail
    render into *render_dir* (recorded as ``record["render"]``) — skipped on
    the CPU backend, where Genesis's Rasterizer camera does not work, and
    never permitted to drop an otherwise-passing fixture.  Passed/dropped
    records are returned; dropped fixtures are removed from the shipped
    catalog by the caller (plan risk #1: drop with a warning, don't fail the
    pipeline).
    """
    import torch

    # Genesis must be initialised before ``unienv_genesis`` is imported: the
    # wrapper's modules reference runtime constants (``gs.qd_float`` …) that
    # only exist once ``gs.init`` has run.
    import genesis as gs

    if getattr(gs, "_initialized", False):
        gs.destroy()
    backend = "cuda" if torch.cuda.is_available() else "cpu"
    gs.init(backend=getattr(gs, backend))

    # Thumbnail pass: Genesis's Rasterizer camera fails on the CPU backend, so
    # rendering is a CUDA-only extra — warn once and degrade to bbox-only.
    render_active = render_dir is not None and backend == "cuda"
    if render_dir is not None and backend != "cuda":
        print(
            f"  [gate] thumbnail render skipped: backend={backend!r} — the "
            f"Rasterizer camera requires CUDA (requested render dir: {render_dir})"
        )

    from unienv_genesis import GenesisWorld
    from unienv_genesis_collection import asset_path
    from unienv_genesis_collection.objects.hub_objects import catalog as catalog_module
    from unienv_genesis_collection.objects.hub_objects.articulated.robocasa_fixture import (
        RoboCasaFixture,
    )

    # The gate runs against the freshly packaged tree: repoint the assets root
    # and drop any cached catalog (per resolved path) from earlier runs.
    asset_path.ASSETS_DIR = output_dir
    catalog_module._CACHE.clear()

    passed: List[dict] = []
    failed: List[dict] = []
    for fixture_id in fixture_ids:
        record: Dict[str, object] = {"fixture_id": fixture_id}
        world = None
        try:
            world = GenesisWorld(scene_init_kwargs={"show_viewer": False})
            descriptor = RoboCasaFixture(fixture_id)
            entity = descriptor.load_into_scene(world, pos=(0.0, 0.0, 0.0))
            world.build_scene(world.scene)

            joints = [
                dict(
                    name=joint.name,
                    type=str(joint.type).split(".")[-1],
                    n_dofs=int(joint.n_dofs),
                    dofs_idx_local=[int(i) for i in joint.dofs_idx_local],
                )
                for joint in entity.joints
            ]
            links = [link.name for link in entity.links]
            record["joints"] = joints
            record["links"] = links
            record["n_joints"] = len(joints)

            spec = FIXTURE_REGISTRY[fixture_id]
            if spec.get("expect_joints") and not joints:
                raise AssertionError(
                    "registry expects an articulated fixture but the built "
                    "entity has no joints"
                )

            # Offline FK union bbox (base frame) → world frame via the pose.
            local_bbox = descriptor.get_local_bounding_box()
            low_local = local_bbox.low.detach().cpu().numpy().astype(float)
            high_local = local_bbox.high.detach().cpu().numpy().astype(float)
            pos = entity.get_pos().detach().cpu().numpy().astype(float)
            quat = entity.get_quat().detach().cpu().numpy().astype(float)
            corners = np.array([
                [sx, sy, sz]
                for sx in (low_local[0], high_local[0])
                for sy in (low_local[1], high_local[1])
                for sz in (low_local[2], high_local[2])
            ])
            world_corners = corners @ quat_to_matrix(quat).T + pos
            low_off = world_corners.min(axis=0)
            high_off = world_corners.max(axis=0)

            v_aabb = entity.get_vAABB()
            low_v = np.asarray(
                v_aabb[0].detach().cpu() if torch.is_tensor(v_aabb[0]) else v_aabb[0],
                dtype=float,
            ).reshape(3)
            high_v = np.asarray(
                v_aabb[1].detach().cpu() if torch.is_tensor(v_aabb[1]) else v_aabb[1],
                dtype=float,
            ).reshape(3)

            extent = np.maximum(high_off - low_off, 1e-6)
            tol = np.maximum(bbox_tol_abs, bbox_tol_rel * extent)
            deltas = np.maximum(
                np.abs(low_v - low_off), np.abs(high_v - high_off)
            )
            record["bbox"] = dict(
                offline_low=low_off.tolist(), offline_high=high_off.tolist(),
                vAABB_low=low_v.tolist(), vAABB_high=high_v.tolist(),
                max_delta=float(deltas.max()), tolerance=float(tol.max()),
            )
            if bool((deltas > tol).any()):
                raise AssertionError(
                    "offline-FK bbox deviates from entity.get_vAABB() beyond "
                    f"tolerance (per-axis deltas {np.round(deltas, 4).tolist()}, "
                    f"tolerance {np.round(tol, 4).tolist()})"
                )

            # All gate signals are gathered: tear the gate scene down before
            # the thumbnail pass builds its own throwaway scene (mirrors the
            # ReplicaCAD gate — one live scene at a time).
            if world is not None:
                try:
                    world.scene.destroy()
                except Exception:  # noqa: BLE001 — teardown is best-effort
                    pass
                world = None

            # Thumbnail render — best-effort, never drops the fixture.  The
            # Rasterizer camera needs the CUDA backend; the whole pass is
            # skipped (with a run-level warning) on CPU.
            record["render"] = None
            if render_active:
                try:
                    record["render"] = _render_thumbnail(
                        descriptor, fixture_id, render_dir,
                    )
                except Exception as exc:  # noqa: BLE001 — keep the passing fixture
                    print(f"  [gate] {fixture_id}: render FAILED (fixture kept) — "
                          f"{type(exc).__name__}: {exc}")

            record["status"] = "passed"
            passed.append(record)
            print(
                f"  [gate] {fixture_id}: OK  joints={len(joints)} "
                f"links={len(links)}  bbox Δ={deltas.max():.4f} m "
                f"(tol {tol.max():.4f} m)"
            )
        except Exception as exc:  # noqa: BLE001 — gate failure ⇒ drop, not crash
            record["status"] = "dropped"
            record["error"] = f"{type(exc).__name__}: {exc}"
            failed.append(record)
            print(f"  [gate] {fixture_id}: DROPPED — {record['error']}")
        finally:
            if world is not None:
                try:
                    world.scene.destroy()
                except Exception:  # noqa: BLE001 — teardown is best-effort
                    pass

    return passed, failed


# ---------------------------------------------------------------------------
# Kitchen arena shells (scenes/rooms/robocasa_kitchen/<variant>/)
# ---------------------------------------------------------------------------

SHELL_TEXTURES = {
    "floor": "textures/tiles/diamond_white_tiles.png",
    "wall": "textures/flat/warm_white.png",
    "counter_top": "textures/marble/marble.png",
    "counter_base": "textures/wood/wood_grain_1.png",
}

SHELL_README = """\
# RoboCasa kitchen arena shells

Static kitchen shells (walls / floor / counter run) generated by
`genesis_adaptor/scripts/prepare_robocasa_assets.py` for the room builder
(`unienv_genesis_collection/scenes/room_builder.py`).  Each variant directory
is self-contained (`model.xml` + textures) and loads with `gs.morphs.MJCF` —
the jointless root welds it to the world (fixed).  The shell origin coincides
with the room frame (floor top at z = 0, room centre at the origin).

Variants:

- `open_kitchen` — 6.0 x 5.0 m room, counter run along the back wall.
- `galley` — 5.0 x 4.0 m room, counter runs along two facing walls.

Textures come from RoboCasa's environment textures (`textures.zip`, CC BY 4.0).
"""

PROVENANCE_README = """\
# RoboCasa kitchen fixtures

Repackaged from the [RoboCasa](https://robocasa.ai/) kitchen asset
distribution (`fixtures.zip` + `textures.zip`, CC BY 4.0 — see `LICENSE`;
obtained from the Hugging Face assets mirror or
`python -m robocasa.scripts.download_kitchen_assets`).

RoboCasa's kitchen fixtures are MJCF models under
`robocasa/models/assets/fixtures/`:

* **Static fixtures** (appliances — dishwashers, fridges, microwaves, ovens,
  stoves, sinks — and the counter modules) ship authored MJCFs (`model.xml` +
  `visuals/` meshes and textures).  They are copied with every referenced
  file — including recursively resolved `<include>`s, mirroring
  `genesis.utils.mjcf.build_model` — with references rewritten
  directory-local, and their **authored collision geoms used as-is**.

* **Composed fixtures** (`cabinet_hinge`, `cabinet_single`, `drawer`,
  `panel`, `handle_*`) do not exist as static MJCFs upstream: RoboCasa
  composes them procedurally at runtime from primitive templates
  (`robocasa/models/fixtures/{cabinets,cabinet_panels,handles}.py`).  This
  packaging ports that composition for a fixed style — **slab panels,
  bar handles, `textures/wood/wood_grain_1.png`** — and bakes the composed
  MuJoCo primitives into exact-dimension mesh geoms so the collection's
  offline MJCF forward-kinematics walker (which reads `<mesh>` geoms only)
  yields per-link visual geometry for the reference-configuration union bbox.
  Collision becomes a convex box/cylinder mesh (identical silhouette);
  reset-region marker geoms are preserved as authored.

Every packaged fixture passed a **Genesis sanity gate**: loaded with
`RoboCasaFixture.load_into_scene` (GPU), non-empty joint set where expected,
and the offline-FK union bbox cross-checked against `entity.get_vAABB()` at
q = 0.  Per-fixture details (discovered joint/link names, bbox deltas, file
lists) are recorded in `packaging_log.json`.

When using these models, please cite RoboCasa:

> S. Nasiriany, A. Maddukuri, L. Zhang, et al.,
> "RoboCasa: A Large-Scale Simulation Framework for Robot Learning", 2024.

Assets © RoboCasa / Lightwheel AI, licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).  Repackaged by
`genesis_adaptor/scripts/prepare_robocasa_assets.py`.
"""


def export_kitchen_shells(source: Path, output_dir: Path) -> List[str]:
    """Write the kitchen arena shells for the room builder (plan §2.1).

    Static MJCF walls/floor/counter shells (visual + collision via plain box
    geoms; jointless root = welded) under
    ``scenes/rooms/robocasa_kitchen/<variant>/`` — the room builder resolves
    ``shell_id="robocasa_kitchen/<variant>"`` to the contained ``model.xml``
    and loads it with ``gs.morphs.MJCF``.
    """
    wall_h = 2.7
    thickness = 0.1
    counter_depth = 0.65
    base_h = 0.87  # cabinet base top
    top_h = 0.05  # countertop thickness

    variants = {
        "open_kitchen": dict(width=6.0, depth=5.0,
                             counters=[(2.4, 5.0 / 2.0 - counter_depth / 2.0)]),
        "galley": dict(
            width=5.0, depth=4.0,
            counters=[
                (2.6, 4.0 / 2.0 - counter_depth / 2.0),
                (2.6, -(4.0 / 2.0 - counter_depth / 2.0)),
            ],
        ),
    }

    asset_lines = [
        f'    <texture file="{SHELL_TEXTURES["floor"]}" type="2d" name="tex_floor"/>',
        f'    <texture file="{SHELL_TEXTURES["wall"]}" type="2d" name="tex_wall"/>',
        f'    <texture file="{SHELL_TEXTURES["counter_top"]}" type="2d" name="tex_marble"/>',
        f'    <texture file="{SHELL_TEXTURES["counter_base"]}" type="2d" name="tex_wood"/>',
        '    <material name="floor_mat" texture="tex_floor" shininess="0.1" texrepeat="8 8" texuniform="true"/>',
        '    <material name="wall_mat" texture="tex_wall" shininess="0.1" texrepeat="4 4" texuniform="true"/>',
        '    <material name="counter_top_mat" texture="tex_marble" shininess="0.3" texrepeat="2 2" texuniform="true"/>',
        '    <material name="counter_base_mat" texture="tex_wood" shininess="0.1" texrepeat="2 2" texuniform="true"/>',
    ]

    out_root = output_dir / "scenes" / "rooms" / "robocasa_kitchen"
    out_root.mkdir(parents=True, exist_ok=True)
    for name, (source_rel,) in [(k, (v,)) for k, v in SHELL_TEXTURES.items()]:
        src = source / source_rel
        if not src.is_file():
            raise FileNotFoundError(f"Shell texture missing in source: {source_rel}")
    for variant, dims in variants.items():
        variant_dir = out_root / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        width, depth = dims["width"], dims["depth"]
        geoms = [
            # Floor: room frame origin at floor centre, floor top at z = 0.
            f'      <geom name="floor" type="box" material="floor_mat" '
            f'size="{_a2s((width / 2.0 + thickness, depth / 2.0 + thickness, thickness / 2.0))}" '
            f'pos="{_a2s((0.0, 0.0, -thickness / 2.0))}"/>',
        ]
        walls = [
            # MuJoCo box geoms take HALF-extents; walls sit just outside the
            # room rect, floor top at z = 0.
            ("north", ((width + 2 * thickness) / 2.0, thickness / 2.0, wall_h / 2.0),
             (0.0, depth / 2.0 + thickness / 2.0, wall_h / 2.0)),
            ("south", ((width + 2 * thickness) / 2.0, thickness / 2.0, wall_h / 2.0),
             (0.0, -(depth / 2.0 + thickness / 2.0), wall_h / 2.0)),
            ("east", (thickness / 2.0, depth / 2.0, wall_h / 2.0),
             (width / 2.0 + thickness / 2.0, 0.0, wall_h / 2.0)),
            ("west", (thickness / 2.0, depth / 2.0, wall_h / 2.0),
             (-(width / 2.0 + thickness / 2.0), 0.0, wall_h / 2.0)),
        ]
        for wall_name, size, pos in walls:
            geoms.append(
                f'      <geom name="wall_{wall_name}" type="box" material="wall_mat" '
                f'size="{_a2s(size)}" pos="{_a2s(pos)}"/>'
            )
        for index, (length, y_centre) in enumerate(dims["counters"]):
            geoms.append(
                f'      <geom name="counter_base_{index}" type="box" material="counter_base_mat" '
                f'size="{_a2s((length / 2.0, counter_depth / 2.0, (base_h - 0.10) / 2.0))}" '
                f'pos="{_a2s((0.0, y_centre, 0.10 + (base_h - 0.10) / 2.0))}"/>'
            )
            geoms.append(
                f'      <geom name="counter_top_{index}" type="box" material="counter_top_mat" '
                f'size="{_a2s(((length + 0.05) / 2.0, (counter_depth + 0.04) / 2.0, top_h / 2.0))}" '
                f'pos="{_a2s((0.0, y_centre, base_h + top_h / 2.0))}"/>'
            )
        mjcf = (
            '<mujoco model="robocasa_kitchen">\n'
            '  <compiler angle="radian"/>\n'
            "  <asset>\n" + "\n".join(asset_lines) + "\n  </asset>\n"
            "  <worldbody>\n"
            '    <body name="room">\n' + "\n".join(geoms) + "\n    </body>\n"
            "  </worldbody>\n"
            "</mujoco>\n"
        )
        for source_rel in SHELL_TEXTURES.values():
            dest = variant_dir / source_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / source_rel, dest)
        (variant_dir / "model.xml").write_text(mjcf)
        compile_check(variant_dir / "model.xml")
        print(f"  shell {variant}: {variant_dir / 'model.xml'}")
    (out_root / "README.md").write_text(SHELL_README)
    (out_root / "LICENSE").write_text(CC_BY_40_TEXT)
    return sorted(variants)


CC_BY_40_TEXT = """\
RoboCasa kitchen fixtures and environment textures are licensed under
CC BY 4.0 by RoboCasa / Lightwheel AI.

This repackaging (directory-local MJCFs, composed fixtures, kitchen shells)
is likewise distributed under CC BY 4.0.

Full license text: https://creativecommons.org/licenses/by/4.0/
"""


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def write_catalog(out_root: Path, records: Sequence[dict]) -> int:
    """Catalog entries: language annotations (broad → detailed) + tags.

    Tags carry ``furniture``/``kitchen``/``articulated`` plus the fixture
    family (``cabinet``/``drawer``/…).  No joint/part metadata is baked — the
    only articulation signal is the ``articulated`` tag, derived from the
    Genesis-gate-observed joint count (ground truth, not the MJCF source).
    Records may carry ``annotations``/``tags`` overrides (used by tests for
    synthetic fixtures); otherwise the registry spec provides them.
    """
    catalog: Dict[str, dict] = {}
    for record in records:
        spec = FIXTURE_REGISTRY.get(record["fixture_id"])
        if spec is None:
            annotations = list(record.get("annotations", []))
            tags = set(record.get("tags", []))
        else:
            annotations = list(spec["annotations"])
            tags = {spec["family"]}
        if not annotations and not tags:
            raise KeyError(
                f"Catalog record for unknown fixture {record['fixture_id']!r} "
                "without annotations/tags override"
            )
        tags = set(tags)
        tags.update(("furniture", "kitchen"))
        if record.get("n_joints"):
            tags.add("articulated")
        catalog[record["fixture_id"]] = {
            "language_annotations": annotations,
            "tags": sorted(tags),
        }
    (out_root / "catalog.json").write_text(
        json.dumps(catalog, indent=1, ensure_ascii=False) + "\n"
    )
    n_articulated = sum(1 for entry in catalog.values() if "articulated" in entry["tags"])
    print(
        f"Wrote {out_root / 'catalog.json'} with {len(catalog)} entries "
        f"({n_articulated} articulated)."
    )
    return len(catalog)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def prepare(
    source: Path,
    output_dir: Path,
    fixture_ids: Sequence[str],
    gate: bool = True,
    bbox_tol_abs: float = BBOX_TOL_ABS,
    bbox_tol_rel: float = BBOX_TOL_REL,
    render_dir: Optional[Path] = None,
) -> Tuple[Path, List[str], List[dict]]:
    """Package the selected fixtures + shells into *output_dir*.

    ``render_dir`` (when the gate runs) receives the per-fixture standing
    thumbnails written by the sanity gate; the written paths land in the
    gate's ``packaging_log.json`` records.
    """
    out_root = output_dir / OUTPUT_ASSETS_SUBDIR
    out_root.mkdir(parents=True, exist_ok=True)

    # Preliminary catalog so the gate's RoboCasaFixture descriptors resolve
    # offline; rewritten with the gate survivors below.
    write_catalog(out_root, [{"fixture_id": fid} for fid in fixture_ids])

    packaged: List[dict] = []
    for fixture_id in fixture_ids:
        spec = FIXTURE_REGISTRY[fixture_id]
        print(f"Packaging {fixture_id} ({spec['kind']}) ...")
        out_dir = out_root / fixture_id
        if spec["kind"] == "composed":
            root = compose_fixture(fixture_id, spec, source)
            out_dir.mkdir(parents=True, exist_ok=True)
            copied_refs = copy_asset_refs(
                root, source, (source / "fixtures" / spec["template"]).parent, out_dir,
            )
            baked = bake_primitive_geoms(root, out_dir)
        else:
            source_dir = source / "fixtures" / spec["source"]
            repkg = Repackager(source, out_root)
            out_dir = repkg.repackage(source_dir / "model.xml", fixture_id)
            root = ET.parse(out_dir / "model.xml").getroot()
            copied_refs = sorted(repkg.copied.values())
            # Static fixtures keep their authored meshes as-is; but a fixture
            # shipping *only* MuJoCo primitives (the counter templates) has no
            # offline-FK visual geometry at all — bake its primitives into
            # exact-dimension mesh geoms like the composed fixtures.
            baked = []
            if root.find("asset/mesh") is None and any(
                geom.get("type") in ("box", "cylinder") for geom in root.iter("geom")
            ):
                baked = bake_primitive_geoms(root, out_dir)
        removed = strip_freejoint(root)
        ET.indent(root)
        (out_dir / "model.xml").write_text(ET.tostring(root, encoding="unicode"))
        collision = ensure_collision_geoms(root, out_dir, fixture_id)
        ET.indent(root)
        (out_dir / "model.xml").write_text(ET.tostring(root, encoding="unicode"))
        patched_meshes = compile_check(out_dir / "model.xml")
        packaged.append(dict(
            fixture_id=fixture_id, kind=spec["kind"],
            source=spec.get("source") or spec.get("template"),
            copied_refs=copied_refs,
            baked_meshes=baked, patched_meshes=patched_meshes,
            removed_freejoints=removed, **collision,
            n_files=len(list(out_dir.rglob("*"))),
        ))
        print(f"  -> {out_dir} (collision: {collision['collision_source']}, "
              f"{collision['n_collision_geoms']} collision geoms)")

    if gate:
        print(f"\nGenesis sanity gate on {len(packaged)} fixtures "
              f"(cabinet_hinge + drawer first — plan risk #1) ...")
        ordered = list(CHEAPEST_SIGNAL) + [
            fid for fid in fixture_ids if fid not in CHEAPEST_SIGNAL
        ]
        gate_records, dropped = sanity_gate(
            output_dir, ordered, bbox_tol_abs=bbox_tol_abs, bbox_tol_rel=bbox_tol_rel,
            render_dir=render_dir,
        )
        kept_ids = [record["fixture_id"] for record in gate_records]
    else:
        gate_records = []
        dropped = []
        kept_ids = list(fixture_ids)

    write_catalog(
        out_root,
        gate_records if gate else [{"fixture_id": fid} for fid in fixture_ids],
    )

    (out_root / "LICENSE").write_text(CC_BY_40_TEXT)
    (out_root / "README.md").write_text(PROVENANCE_README)
    (out_root / "packaging_log.json").write_text(json.dumps(dict(
        packaged=packaged,
        gate=gate_records,
        dropped=dropped,
    ), indent=1, ensure_ascii=False) + "\n")

    variants = export_kitchen_shells(source, output_dir)
    print(f"Exported kitchen shells: {variants}")

    print(f"\nPrepared {len(kept_ids)} fixtures in {out_root}")
    if dropped:
        print(f"Dropped {len(dropped)} fixture(s) failing the Genesis sanity gate:")
        for record in dropped:
            print(f"  - {record['fixture_id']}: {record.get('error')}")
    return out_root, kept_ids, dropped


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source", type=Path, default=None,
        help="Local RoboCasa assets download — the robocasa/models/assets tree "
             f"produced by {DOWNLOAD_KITCHEN_ASSETS_DOC}, or any directory "
             "holding fixtures/ + textures/ (or their zips). If omitted, the "
             "assets are fetched with huggingface_hub.snapshot_download from "
             "--repo-id.",
    )
    parser.add_argument(
        "--repo-id", default=DEFAULT_HF_REPO,
        help=f"Hugging Face dataset repo providing fixtures.zip + textures.zip "
             f"(default {DEFAULT_HF_REPO!r}; the official {OFFICIAL_HF_REPO!r} "
             f"publishes only the accessories families in its fixtures.zip).",
    )
    parser.add_argument(
        "--fixtures", default="all",
        help="Comma-separated fixture ids and/or groups ("
             + ", ".join(FIXTURE_GROUPS) + ", all). Default: all.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to place the prepared 'objects/robocasa' and "
             "'scenes/rooms/robocasa_kitchen' trees in.",
    )
    parser.add_argument(
        "--no-gate", action="store_true",
        help="Skip the Genesis sanity-load gate (offline packaging only).",
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="Skip the per-fixture standing render (the gate still checks loads + bboxes).",
    )
    parser.add_argument(
        "--render-dir", type=Path, default=Path("/tmp/robocasa_sanity"),
        help="Directory for the per-fixture eyeball renders (default /tmp/robocasa_sanity).",
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

    fixture_ids = resolve_fixture_selection(args.fixtures.split(","))
    print(f"Fixture selection ({len(fixture_ids)}): {fixture_ids}")

    scratch = Path(tempfile.mkdtemp(prefix="robocasa_assets_"))
    source = resolve_source(args.source, args.repo_id, scratch)
    print(f"Source: {source}")

    out_root, kept, dropped = prepare(
        source,
        args.output_dir.expanduser().resolve(),
        fixture_ids,
        gate=not args.no_gate,
        bbox_tol_abs=args.bbox_tol_abs,
        bbox_tol_rel=args.bbox_tol_rel,
        render_dir=None if args.no_render else args.render_dir,
    )

    print(
        "\nTo upload (requires write access to the org):\n"
        f"  hf upload UniEnvOrg/unienv_genesis_collection_assets \\\n"
        f"      {out_root} objects/robocasa --repo-type dataset\n"
        f"  hf upload UniEnvOrg/unienv_genesis_collection_assets \\\n"
        f"      {args.output_dir.expanduser().resolve() / SHELLS_SUBDIR} "
        f"{SHELLS_SUBDIR} --repo-type dataset"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
