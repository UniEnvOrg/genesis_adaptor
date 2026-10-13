"""The Genesis sanity gate, once (was four diverging ``sanity_check_object`` +
``run_sanity_gate`` copies in the packagers).

The two ``sanity_check_*`` bodies below are the molmospaces/lightwheel copies
line-for-line plus the two strictly-better artvip improvements, now propagated
to every family:

* the pxr weak-proxy guard comment + bound stage (``usd_stage``) — iterating a
  transient ``Usd.Stage.Open(...).Traverse()`` drops the stage mid-walk;
* the ``runtime = item._to_runtime_name(clean, "joint")`` binding in the
  halfway-opened loop (pins the clean→runtime resolution alongside
  ``resolve_joint_dofs``).

``run_sanity_gate`` is the shared loop; the only per-family inputs are the
``descriptor_factory`` (object id → descriptor) and the ``check`` callable —
lightwheel's former static/articulated fork inside the loop collapses to a
10-line factory in its packager.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from _packaging.gate_common import (
    _WorldStub,
    _opened_reference_joints,
    _render_thumbnail,
    _finite_midpoint,
    gs_joint_type_name,
)

# Bbox cross-check tolerances (offline-FK box proxy vs real mesh geometry).
BBOX_TOL_ABS = 0.03  # metres
BBOX_TOL_REL = 0.12  # fraction of the per-axis extent


# ---------------------------------------------------------------------------
# Articulated gate (artvip / lightwheel furniture / molmospaces)
# ---------------------------------------------------------------------------

def sanity_check_articulated(
    item,
    *,
    bbox_tol_abs: float = BBOX_TOL_ABS,
    bbox_tol_rel: float = BBOX_TOL_REL,
    require_moving_joints: bool = True,
    render_dir: Optional[Path] = None,
) -> dict:
    """Genesis sanity-load one packaged object.

    Loads through the descriptor in a throwaway scene (single entity via
    ``add_stage``), asserts a non-empty moving-joint set (the packaged
    stages carry the joint-body swap — this is where an unswapped stage
    fails — *require_moving_joints=True*; the rigid-prop variants turn this
    assert off), duplicate-upstream-leaf-name contract, unique link leaf
    names, and cross-checks the offline box-proxy FK union bbox against
    ``entity.get_vAABB()`` at q = 0 and at a **halfway-opened**
    configuration.  A thumbnail render is best-effort (never drops).

    Returns the gate record; raises on failure.
    """
    import torch as _torch
    import genesis as gs
    from pxr import Usd, UsdPhysics

    from unienv_genesis_collection.objects.articulated_object import (
        usd_link_meshes_at_config,
    )

    # Upstream leaf names — the duplicate check must see *raw* leaves, not
    # the descriptor's resolution to suffix names.  (The stage is bound to a
    # variable: iterating a transient `Usd.Stage.Open(...).Traverse()` drops
    # the stage mid-walk — pxr prims are weak proxies of a dead stage.
    usd_stage = Usd.Stage.Open(str(item.description_file))
    leaf_names = [
        prim.GetName()
        for prim in usd_stage.Traverse()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    dupes = sorted({name for name in leaf_names if leaf_names.count(name) > 1})
    if dupes:
        raise AssertionError(
            f"duplicate link/joint leaf names upstream: {dupes} — these would "
            "ship suffix-disambiguated clean names, which are asset-layout-"
            "derived and not stable for task configs; renaming in packaging "
            "(or an upstream fix) must happen first."
        )

    local = item.get_local_bounding_box()
    low = local.low.detach().cpu().numpy().astype(float)
    high = local.high.detach().cpu().numpy().astype(float)

    scene = gs.Scene(show_viewer=False, renderer=gs.options.renderers.Rasterizer())
    try:
        entity = item.load_into_scene(_WorldStub(scene), pos=(0.0, 0.0, 0.0))
        scene.build(n_envs=0)
        joint_records = [
            dict(
                name=str(joint.name),
                type=gs_joint_type_name(joint.type),
                n_dofs=int(joint.n_dofs),
            )
            for joint in entity.joints
        ]
        moving = [j for j in joint_records if j["n_dofs"] > 0]
        if require_moving_joints and not moving:
            raise AssertionError(
                "no moving (revolute/prismatic) joints after load "
                f"(all joints: {joint_records}) — the joint-body swap may be "
                "missing or the object is not articulated."
            )
        link_paths = [str(link.name) for link in entity.links]
        vaabb = entity.get_vAABB()
        low_v = np.asarray(
            vaabb[0].detach().cpu() if _torch.is_tensor(vaabb[0]) else vaabb[0],
            dtype=float,
        ).reshape(3)
        high_v = np.asarray(
            vaabb[1].detach().cpu() if _torch.is_tensor(vaabb[1]) else vaabb[1],
            dtype=float,
        ).reshape(3)
    finally:
        try:
            scene.destroy()
        except Exception:  # noqa: BLE001 — teardown is best-effort
            pass

    extent = np.maximum(high - low, 1e-6)
    tol = np.maximum(bbox_tol_abs, bbox_tol_rel * extent)
    deltas = np.maximum(np.abs(low_v - low), np.abs(high_v - high))
    record = dict(
        closed=dict(
            offline_low=low.tolist(), offline_high=high.tolist(),
            vAABB_low=low_v.tolist(), vAABB_high=high_v.tolist(),
            max_delta=float(deltas.max()), tolerance_per_axis=tol.tolist(),
        ),
    )
    if bool((deltas > tol).any()):
        raise AssertionError(
            "offline-FK bbox deviates from entity.get_vAABB() beyond tolerance "
            f"(per-axis deltas {np.round(deltas, 4).tolist()}, tolerance "
            f"{np.round(tol, 4).tolist()})"
        )

    # ---- halfway-opened cross-check ---------------------------------------
    # (needs moving joints to open; without the moving-joint requirement the
    # closed-pose record above is the whole comparison)
    if not require_moving_joints:
        rendered: Optional[str] = None
        if render_dir is not None:
            try:
                rendered = _render_thumbnail(item, item.name, render_dir)
            except Exception as exc:  # noqa: BLE001 — keep the passing object
                print(
                    f"  [gate] {item.name}: render FAILED (object kept) — "
                    f"{type(exc).__name__}: {exc}"
                )
        return dict(
            status="passed",
            links=link_paths,
            joints=joint_records,
            n_joints=len(moving),
            clean_joint_names=[item.clean_name_of(j["name"]) for j in moving],
            clean_link_names=sorted(item.list_link_names()),
            bbox=record,
            render=rendered,
        )

    opened = gs.Scene(show_viewer=False, renderer=gs.options.renderers.Rasterizer())
    try:
        opened_entity = item.load_into_scene(_WorldStub(opened), pos=(0.0, 0.0, 0.0))
        opened.build(n_envs=0)
        reference_qpos: Dict[str, float] = {}
        for joint_record in _opened_reference_joints(moving):
            clean = item.clean_name_of(joint_record["name"])
            # Fail-fast: the clean→runtime name must resolve on the stage
            # (the artvip copy pinned this; `_to_runtime_name` raises a
            # descriptive ArticulationError for an unmapped clean name).
            runtime = item._to_runtime_name(clean, "joint")
            dofs_idx = item.resolve_joint_dofs(opened_entity, clean)
            q_min, q_max = opened_entity.get_dofs_limit(dofs_idx_local=dofs_idx)
            midpoint = _finite_midpoint(q_min, q_max)
            item.set_dofs_position(opened_entity, midpoint, clean)
            reference_qpos[clean] = float(midpoint.detach().cpu().item())
        vaabb_open = opened_entity.get_vAABB()
        low_open_v = np.asarray(
            vaabb_open[0].detach().cpu() if _torch.is_tensor(vaabb_open[0]) else vaabb_open[0],
            dtype=float,
        ).reshape(3)
        high_open_v = np.asarray(
            vaabb_open[1].detach().cpu() if _torch.is_tensor(vaabb_open[1]) else vaabb_open[1],
            dtype=float,
        ).reshape(3)
    finally:
        try:
            opened.destroy()
        except Exception:  # noqa: BLE001 — teardown is best-effort
            pass

    opened_meshes = usd_link_meshes_at_config(item.description_file, reference_qpos)
    if not opened_meshes:
        raise AssertionError("no link meshes at the opened configuration")
    low_open = np.min([mesh.bounds[0] for mesh in opened_meshes.values()], axis=0)
    high_open = np.max([mesh.bounds[1] for mesh in opened_meshes.values()], axis=0)
    extent_open = np.maximum(high_open - low_open, 1e-6)
    tol_open = np.maximum(bbox_tol_abs, bbox_tol_rel * extent_open)
    deltas_open = np.maximum(
        np.abs(low_open_v - low_open), np.abs(high_open_v - high_open)
    )
    record["opened"] = dict(
        max_delta=float(deltas_open.max()),
        tolerance=float(tol_open.max()),
        reference_qpos=reference_qpos,
    )
    if bool((deltas_open > tol_open).any()):
        raise AssertionError(
            "offline-FK bbox at the halfway-opened configuration deviates from "
            "entity.get_vAABB() beyond tolerance (per-axis deltas "
            f"{np.round(deltas_open, 4).tolist()}, tolerance "
            f"{np.round(tol_open, 4).tolist()})"
        )

    rendered: Optional[str] = None
    if render_dir is not None:
        try:
            rendered = _render_thumbnail(item, item.name, render_dir)
        except Exception as exc:  # noqa: BLE001 — keep the passing object
            print(
                f"  [gate] {item.name}: render FAILED (object kept) — "
                f"{type(exc).__name__}: {exc}"
            )

    return dict(
        status="passed",
        links=link_paths,
        joints=joint_records,
        n_joints=len(moving),
        clean_joint_names=[item.clean_name_of(j["name"]) for j in moving],
        clean_link_names=sorted(item.list_link_names()),
        bbox=record,
        render=rendered,
    )


# ---------------------------------------------------------------------------
# Rigid-prop gate (simready warehouse) — the closed-bbox + render core of the
# articulated check, minus the articulated-only contracts (the pxr leaf-name
# walk, the moving-joint assert, the halfway-opened cross-check)
# ---------------------------------------------------------------------------

def sanity_check_rigid(
    item,
    *,
    bbox_tol_abs: float = BBOX_TOL_ABS,
    bbox_tol_rel: float = BBOX_TOL_REL,
    render_dir: Optional[Path] = None,
) -> dict:
    """Genesis sanity-load one packaged prop.

    Adapted from the ArtVIP gate for **rigid props**: the moving-joint set
    may be empty (nothing is dropped for it), but the single-entity assert,
    the offline box-proxy FK union bbox vs ``entity.get_vAABB()`` and the
    thumbnail render all still apply.
    """
    import torch as _torch
    import genesis as gs

    local = item.get_local_bounding_box()
    low = local.low.detach().cpu().numpy().astype(float)
    high = local.high.detach().cpu().numpy().astype(float)

    scene = gs.Scene(show_viewer=False, renderer=gs.options.renderers.Rasterizer())
    try:
        entity = item.load_into_scene(_WorldStub(scene), pos=(0.0, 0.0, 0.0))
        scene.build(n_envs=0)
        joint_records = [
            dict(
                name=str(joint.name),
                type=gs_joint_type_name(joint.type),
                n_dofs=int(joint.n_dofs),
            )
            for joint in entity.joints
        ]
        moving = [j for j in joint_records if j["n_dofs"] > 0]
        # RIGID PROPS: an empty moving-joint set is the *expected* shape —
        # nothing is dropped for it.  The prop must still have a body.
        if not entity.links:
            raise AssertionError("no links after load (a broken/empty stage)")
        link_paths = [str(link.name) for link in entity.links]
        vaabb = entity.get_vAABB()
        low_v = np.asarray(
            vaabb[0].detach().cpu() if _torch.is_tensor(vaabb[0]) else vaabb[0],
            dtype=float,
        ).reshape(3)
        high_v = np.asarray(
            vaabb[1].detach().cpu() if _torch.is_tensor(vaabb[1]) else vaabb[1],
            dtype=float,
        ).reshape(3)
    finally:
        try:
            scene.destroy()
        except Exception:  # noqa: BLE001 — teardown is best-effort
            pass

    extent = np.maximum(high - low, 1e-6)
    tol = np.maximum(bbox_tol_abs, bbox_tol_rel * extent)
    deltas = np.maximum(np.abs(low_v - low), np.abs(high_v - high))
    record = dict(
        closed=dict(
            offline_low=low.tolist(), offline_high=high.tolist(),
            vAABB_low=low_v.tolist(), vAABB_high=high_v.tolist(),
            max_delta=float(deltas.max()), tolerance_per_axis=tol.tolist(),
        ),
    )
    if bool((deltas > tol).any()):
        raise AssertionError(
            "offline-FK bbox deviates from entity.get_vAABB() beyond tolerance "
            f"(per-axis deltas {np.round(deltas, 4).tolist()}, tolerance "
            f"{np.round(tol, 4).tolist()})"
        )

    rendered: Optional[str] = None
    if render_dir is not None:
        try:
            rendered = _render_thumbnail(item, item.name, render_dir)
        except Exception as exc:  # noqa: BLE001 — keep the passing prop
            print(
                f"  [gate] {item.name}: render FAILED (object kept) — "
                f"{type(exc).__name__}: {exc}"
            )

    return dict(
        status="passed",
        links=link_paths,
        joints=joint_records,
        n_moving_joints=len(moving),
        clean_joint_names=[],
        clean_link_names=sorted(item.list_link_names()),
        bbox=record,
        render=rendered,
    )


# ---------------------------------------------------------------------------
# Shared gate loop
# ---------------------------------------------------------------------------

def _prepare_offline_assets(output_dir: Path, asset_noun: str = "object"):
    """Repoint ``asset_path`` at the freshly packaged tree; stub downloads.

    The gate must run from the freshly packaged tree only (a Hub download
    during packaging would mask an incomplete repackage).
    """
    from unienv_genesis_collection import asset_path
    from unienv_genesis_collection.objects.hub_objects import catalog as catalog_module

    original_assets_dir = asset_path.ASSETS_DIR
    original_download = asset_path.download_assets

    def _no_download(**kwargs):
        raise RuntimeError(
            f"Unexpected Hub download during the sanity gate ({kwargs}); "
            f"the repackaged {asset_noun} directory is incomplete."
        )

    asset_path.download_assets = _no_download
    asset_path.ASSETS_DIR = output_dir
    catalog_module._CACHE.clear()
    return original_assets_dir, original_download


def _restore_offline_assets(original_assets_dir, original_download) -> None:
    """Undo :func:`_prepare_offline_assets`'s global patching (``finally``)."""
    from unienv_genesis_collection import asset_path
    from unienv_genesis_collection.objects.hub_objects import catalog as catalog_module

    asset_path.download_assets = original_download
    asset_path.ASSETS_DIR = original_assets_dir
    catalog_module._CACHE.clear()


def _init_genesis(render_dir: Optional[Path]) -> Tuple[bool, str]:
    """Boot Genesis (CUDA when available, else CPU) for the gate; reports
    whether the thumbnail render is active on that backend."""
    import torch as _torch

    import genesis as gs

    if getattr(gs, "_initialized", False):
        gs.destroy()
    backend = "cuda" if _torch.cuda.is_available() else "cpu"
    gs.init(backend=getattr(gs, backend))

    render_active = render_dir is not None and backend == "cuda"
    if render_dir is not None and backend != "cuda":
        # Thumbnail renders need the CUDA backend (Rasterizer camera).
        print(
            f"  [gate] thumbnail render skipped: backend={backend!r} — the "
            f"Rasterizer camera requires CUDA (render dir {render_dir})"
        )
    return render_active, backend


def _gate_ok_line(record: dict) -> str:
    if record.get("family") == "static":
        return (
            f"  [gate] {record['object_id']}: OK  static  "
            f"colliders={record['n_colliders']}  "
            f"entities={record['n_entities']}  "
            f"bbox Δ={record['bbox']['max_delta']:.4f} m "
            f"(tol {max(record['bbox']['tolerance_per_axis']):.4f} m)"
        )
    if "n_moving_joints" in record:
        return (
            f"  [gate] {record['object_id']}: OK  links={len(record['links'])} "
            f"moving_joints={record['n_moving_joints']}  "
            f"bbox Δ={record['bbox']['closed']['max_delta']:.4f} m "
            f"(tol {max(record['bbox']['closed']['tolerance_per_axis']):.4f} m)"
        )
    return (
        f"  [gate] {record['object_id']}: OK  joints={record['n_joints']}/{len(record['joints'])} "
        f"links={len(record['links'])}  bbox Δ={record['bbox']['closed']['max_delta']:.4f} m "
        f"(tol {max(record['bbox']['closed']['tolerance_per_axis']):.4f} m)"
        + (
            f"  opened Δ={record['bbox']['opened']['max_delta']:.4f} m"
            if record["bbox"].get("opened")
            else ""
        )
    )


def run_sanity_gate(
    output_dir: Path,
    object_ids: Sequence[str],
    *,
    descriptor_factory: Callable[[str], object],
    check: Callable[..., dict] = sanity_check_articulated,
    render_dir: Optional[Path] = None,
    bbox_tol_abs: float = BBOX_TOL_ABS,
    bbox_tol_rel: float = BBOX_TOL_REL,
    pre_check: Optional[Callable[[str, object], Optional[dict]]] = None,
    asset_noun: str = "object",
) -> Tuple[List[str], List[dict], List[dict]]:
    """Sanity-load every packaged object; drop (warn on) the failing ones.

    ``descriptor_factory(object_id)`` builds the per-object descriptor (the
    family hub class — the lightwheel two-family dispatch also lives in its
    factory, which returns ``(item, check)`` pairs); *check* is the family
    gate used when the factory returns a bare item, and *pre_check* is an
    optional per-object guard after the descriptor resolves (artvip's A7
    oversized-asset extent check) that returns a drop record or ``None``.

    Both descriptor construction and the *check* call are inside the
    per-object guard: an unknown id or a failing object is a per-object
    drop, never a batch kill.

    Runs from the freshly packaged tree only (a Hub download during
    packaging would mask an incomplete repackage — assets are repointed at
    *output_dir* and downloads are stubbed out).  Returns
    ``(kept_object_ids, passed_records, dropped_records)``.
    """
    original_assets_dir, original_download = _prepare_offline_assets(
        output_dir, asset_noun
    )
    try:
        render_active, _backend = _init_genesis(render_dir)

        kept: List[str] = []
        passed_records: List[dict] = []
        dropped_records: List[dict] = []
        for object_id in object_ids:
            # Inside the per-object guard: an id missing from the catalog
            # (the packaged tree's catalog only lists survivors) or an
            # otherwise unloadable descriptor is recorded as that object's
            # drop instead of killing the whole batch worker (and with it
            # every other object's record).
            try:
                built = descriptor_factory(object_id)
                if isinstance(built, tuple) and len(built) == 2 and callable(built[1]):
                    item, check_fn = built
                else:
                    item, check_fn = built, check
            except Exception as exc:  # noqa: BLE001 — drop + warn, never fail
                dropped_records.append(
                    dict(object_id=object_id, status="dropped",
                         error=f"{type(exc).__name__}: {exc}")
                )
                print(f"  [gate] {object_id}: DROPPED — {type(exc).__name__}: {exc}")
                continue
            if pre_check is not None:
                # The hook prints its own DROPPED line (artvip's oversized
                # record carries a distinct "OversizedAsset:" error).
                drop_record = pre_check(object_id, item)
                if drop_record is not None:
                    dropped_records.append(drop_record)
                    continue
            try:
                record = check_fn(
                    item,
                    bbox_tol_abs=bbox_tol_abs,
                    bbox_tol_rel=bbox_tol_rel,
                    render_dir=render_dir if render_active else None,
                )
            except Exception as exc:  # noqa: BLE001 — drop + warn, never fail
                dropped_records.append(
                    dict(object_id=object_id, status="dropped",
                         error=f"{type(exc).__name__}: {exc}")
                )
                print(f"  [gate] {object_id}: DROPPED — {type(exc).__name__}: {exc}")
                continue
            record["object_id"] = object_id
            passed_records.append(record)
            kept.append(object_id)
            print(_gate_ok_line(record))
        return kept, passed_records, dropped_records
    finally:
        _restore_offline_assets(original_assets_dir, original_download)
