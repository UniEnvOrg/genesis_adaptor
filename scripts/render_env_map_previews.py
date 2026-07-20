#!/usr/bin/env python3
"""Render one preview image per downloaded HDRI for visual curation.

Walks every ``*.hdr`` file in
``unienv_genesis_collection/assets/scenes/env_maps``, builds a lightweight
Genesis/UniEnv scene (table + Franka FR3 robot + one Nyx RGB camera) lit
**only** by the HDRI, captures a single RGB frame, and saves it as a PNG into
a ``previews/`` subfolder of the env_maps directory.

A JSON manifest (``info.json``) is maintained next to the HDRs.  For each
HDRI it records:

* ``hdr_path``       – relative path to the .hdr file (relative to env_maps/)
* ``preview_path``   – relative path to the rendered preview PNG
* ``status``         – ``"pending"`` (default) / ``"approved"`` / ``"rejected"`` /
                       ``"failed"`` (auto, reset on next successful render);
                       user-editable between runs
* ``rotation_deg``   – user-editable rotation hint (default 0.0)
* ``notes``          – free-form user notes (default empty)
* ``rendered_at``    – ISO-8601 timestamp of the most recent render
* ``_last_error``    – (only when a render failed) auto-generated error message

Re-runs preserve any user-edited ``status`` / ``rotation_deg`` / ``notes``
fields.  The auto-generated ``"failed"`` status and ``_last_error`` are
cleared the next time the HDRI renders successfully.  Pass ``--force`` to
re-render previews that already exist.

The script is intentionally small and self-contained: it does not touch any
files outside ``assets/scenes/env_maps/`` and depends only on the existing
``unienv_genesis`` / ``unienv_genesis_collection`` packages, plus Pillow (or
imageio as a fallback) for PNG writing.

Usage examples
--------------

Render every HDRI that does not already have a preview::

    python scripts/render_env_map_previews.py

Preview just the first five HDRs (after sorting)::

    python scripts/render_env_map_previews.py --limit 5

Force a re-render of every HDRI, even if a preview already exists::

    python scripts/render_env_map_previews.py --force

Render only HDRs whose filename matches a glob (case-insensitive)::

    python scripts/render_env_map_previews.py --pattern "*studio*"

Use the side camera view instead of the front view::

    python scripts/render_env_map_previews.py --camera side

Lower the per-pixel sample count for faster previews::

    python scripts/render_env_map_previews.py --spp 32 --res 640x360

List what would be rendered without actually rendering (no Genesis needed)::

    python scripts/render_env_map_previews.py --dry-run
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

# Resolve the env_maps asset directory relative to this script.  This script
# lives at ``genesis_adaptor/scripts/`` so the collection's assets are two
# directories up under ``unienv_genesis_collection/assets/``.
_THIS_DIR = Path(__file__).resolve().parent
_COLLECTION_DIR = _THIS_DIR.parent / "unienv_genesis_collection"
_DEFAULT_ENV_MAP_DIR = _COLLECTION_DIR / "assets" / "scenes" / "env_maps"
_DEFAULT_PREVIEW_SUBDIR = "previews"
_DEFAULT_MANIFEST = "info.json"

# Default table / robot / camera layout (mirrors
# ``examples/pickplace_fr3_robotiq.py`` so the previews show the same
# arrangement users see in the pickplace example).
TABLE_SURFACE_Z = 0.6
TABLE_YAW = math.pi / 2
ROBOT_POS = (-0.45, 0.0, TABLE_SURFACE_Z)

# Camera presets: (name, (pos, lookat)).  Front faces the robot from +X; side
# looks at the scene from +Y for a perpendicular view of the table.
CAMERA_PRESETS: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = {
    "front": ((1.0, 0.0, 1.4), (0.0, 0.0, 0.6)),
    "side": ((0.0, 1.2, 1.0), (0.0, 0.0, 0.6)),
}

# Schema version of ``info.json``; bump if we change the manifest shape.
_MANIFEST_SCHEMA_VERSION = 1

# Preserved user-editable fields.  When re-rendering an existing entry we keep
# these untouched; only ``rendered_at`` and ``preview_path`` get refreshed.
# (Note: ``status`` is *partially* preserved — see :func:`_build_entry` for
# the special case where a previous auto-failure is reset on a successful
# re-render.)
_PRESERVED_FIELDS = ("notes", "rotation_deg")
# Field where the most recent auto-generated render error (if any) is stored.
# It is overwritten (not appended) on every render and is *not* user-editable.
_LAST_ERROR_FIELD = "_last_error"
# Status value the script sets when a render fails.  It is treated as
# auto-generated and is reset to ``"pending"`` on the next successful render.
_AUTO_FAILED_STATUS = "failed"


# ---------------------------------------------------------------------------
# Image writing
# ---------------------------------------------------------------------------


def _save_rgb_png(rgb: "object", dest: Path) -> None:
    """Save a ``(H, W, 3)`` uint8 RGB array as a PNG via PIL or imageio.

    Accepts both torch tensors and numpy arrays.  Falls back to imageio if
    Pillow is unavailable.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Convert to numpy (H, W, 3) uint8, contiguous.
    import numpy as np
    import torch

    if hasattr(rgb, "detach"):  # torch.Tensor
        arr_t = rgb.detach().to("cpu")
        if arr_t.dtype != torch.uint8:
            # Floating-point frames: scale to [0, 255] uint8.
            arr_t = arr_t.clamp(0, 255).to(torch.uint8)
        arr = arr_t.contiguous().numpy()
    else:
        arr = np.asarray(rgb)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        arr = np.ascontiguousarray(arr)

    try:
        from PIL import Image
        Image.fromarray(arr).save(dest, format="PNG")
        return
    except ImportError:
        pass

    # Fallback: imageio.  Slower startup but no PIL dependency.
    try:
        import imageio.v2 as imageio
    except ImportError:  # pragma: no cover
        import imageio  # type: ignore[no-redef]
    imageio.imwrite(dest, arr, format=".png")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _load_manifest(manifest_path: Path) -> dict:
    """Load the curation manifest from disk or return an empty skeleton."""
    if manifest_path.is_file():
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError(
                    f"Manifest root must be an object, got {type(data).__name__}"
                )
            return data
        except (json.JSONDecodeError, ValueError) as e:
            print(
                f"  WARNING: could not parse existing manifest {manifest_path} "
                f"({e}); starting from a fresh one.",
                file=sys.stderr,
            )
    return {
        "_schema_version": _MANIFEST_SCHEMA_VERSION,
        "_generated_at": _now_iso(),
        "entries": {},
    }


def _save_manifest(manifest_path: Path, manifest: dict) -> None:
    """Atomically write the manifest to disk (pretty-printed JSON)."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(manifest_path)


def _now_iso() -> str:
    """Return the current UTC time in ISO-8601 form with a ``Z`` suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_entry(
    *,
    stem: str,
    hdr_relpath: str,
    preview_relpath: str,
    existing: Optional[dict],
    rendered_at: str,
    status: str = "pending",
    last_error: Optional[str] = None,
) -> dict:
    """Construct a manifest entry, preserving any existing user edits.

    ``notes`` and ``rotation_deg`` are taken from the existing entry when
    present.  ``status`` is preserved as long as the previous value is *not*
    the auto-generated ``"failed"`` value (which is reset to ``"pending"`` on
    a successful re-render).  ``rendered_at`` and ``preview_path`` are
    always refreshed; ``hdr_path`` is re-derived from the stem.  The
    most recent render error (if any) is stored in ``_last_error``.
    """
    entry: dict = {
        "hdr_path": hdr_relpath,
        "preview_path": preview_relpath,
        "status": status,
        "rotation_deg": 0.0,
        "notes": "",
        "rendered_at": rendered_at,
    }
    if existing:
        for key in _PRESERVED_FIELDS:
            if key in existing:
                entry[key] = existing[key]
        # Preserve the user's status (e.g. "approved" / "rejected"), but
        # treat the auto-generated "failed" status as a one-shot signal: if
        # the new render succeeded, drop it so the entry becomes "pending"
        # again rather than getting stuck on a stale failure.
        prev_status = existing.get("status")
        if prev_status in (None, _AUTO_FAILED_STATUS):
            entry["status"] = status
        else:
            entry["status"] = prev_status
    entry["hdr_path"] = hdr_relpath
    entry["preview_path"] = preview_relpath
    entry["rendered_at"] = rendered_at
    if last_error is not None:
        entry[_LAST_ERROR_FIELD] = last_error
    return entry


# ---------------------------------------------------------------------------
# Scene construction (deferred imports)
# ---------------------------------------------------------------------------


def _parse_resolution(spec: str) -> Tuple[int, int]:
    """Parse a ``WIDTHxHEIGHT`` spec into ``(width, height)``."""
    try:
        w_str, h_str = spec.lower().split("x", 1)
        width, height = int(w_str), int(h_str)
    except (ValueError, AttributeError) as e:
        raise argparse.ArgumentTypeError(
            f"--res must be of the form WIDTHxHEIGHT (e.g. 1280x720), got {spec!r}"
        ) from e
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError(
            f"--res dimensions must be positive, got {width}x{height}"
        )
    return width, height


def _resolve_backend(gs_backend, name: str):
    """Map a CLI backend name to the genesis.backend enum value."""
    name = name.lower()
    if name == "gpu":
        return gs_backend.gpu
    if name == "cpu":
        return gs_backend.cpu
    if name == "metal":
        return gs_backend.metal
    raise argparse.ArgumentTypeError(
        f"Unknown backend {name!r}; expected 'cpu', 'gpu' or 'metal'."
    )


def _init_genesis(backend_name: str, precision: str) -> "object":
    """Initialise Genesis (idempotent) and return the ``genesis`` module.

    Importing the UniEnv/Genesis adaptor packages requires a fully initialised
    ``gs`` module, so the caller should invoke :func:`_init_genesis` before
    importing any of them.
    """
    import genesis as gs
    from genesis.constants import backend as gs_backend

    if not getattr(gs, "_initialized", False):
        gs.init(
            backend=_resolve_backend(gs_backend, backend_name),
            precision=precision,
            logging_level="warning",
        )
    return gs


def _import_adaptors():
    """Import the UniEnv/Genesis adaptor packages.

    Must be called *after* :func:`_init_genesis` so the ``gs`` namespace is
    fully populated (some adaptor submodules touch ``gs.qd_float`` at import
    time).
    """
    from unienv_genesis import GenesisWorld
    from unienv_genesis.nodes.sensors.camera_node import (
        CameraSensorNode,
        CameraSensorNodeOptions,
    )
    from unienv_genesis.options.light_option import GenesisWorldEnvMapOption
    from unienv_genesis_collection.robots.franka_fr3_robotiq_2f85 import (
        FrankaFR3WithRobotiq2F85,
    )
    from unienv_genesis_collection.scenes.table_builder import TableBuilder
    from unienv_interface.world import WorldEnv
    return {
        "GenesisWorld": GenesisWorld,
        "CameraSensorNode": CameraSensorNode,
        "CameraSensorNodeOptions": CameraSensorNodeOptions,
        "GenesisWorldEnvMapOption": GenesisWorldEnvMapOption,
        "FrankaFR3WithRobotiq2F85": FrankaFR3WithRobotiq2F85,
        "TableBuilder": TableBuilder,
        "WorldEnv": WorldEnv,
    }


def _pick_fixed_table_texture() -> Optional[Path]:
    """Pick a stable texture so the table looks the same across previews.

    Falls back to a random texture if the table assets have not been
    downloaded yet (TableBuilder will pick at random and we re-pick
    deterministically the next time assets are present).
    """
    tex_dir = _COLLECTION_DIR / "assets" / "scenes" / "table" / "tabletop_textures"
    if not tex_dir.is_dir():
        return None
    wood = sorted(tex_dir.glob("wood_texture_*.jpg"))
    if wood:
        return wood[len(wood) // 2]  # middle one for a balanced look
    fallbacks = sorted(tex_dir.glob("*.jpg"))
    return fallbacks[0] if fallbacks else None


def _build_world(
    *,
    backend_name: str,
    spp: int,
    precision: str,
    camera: str,
    res: Tuple[int, int],
):
    """Build the static ``WorldEnv`` reused for every HDRI preview.

    Returns ``(env, world, cam_node, env_map_dir)``.  The caller mutates
    ``world.env_map`` and calls ``env.reset(reload=True)`` for each HDRI.
    """
    gs = _init_genesis(backend_name, precision)
    adaptors = _import_adaptors()
    GenesisWorld = adaptors["GenesisWorld"]
    CameraSensorNode = adaptors["CameraSensorNode"]
    CameraSensorNodeOptions = adaptors["CameraSensorNodeOptions"]
    FrankaFR3WithRobotiq2F85 = adaptors["FrankaFR3WithRobotiq2F85"]
    TableBuilder = adaptors["TableBuilder"]
    WorldEnv = adaptors["WorldEnv"]

    env_map_dir = _DEFAULT_ENV_MAP_DIR
    env_map_dir.mkdir(parents=True, exist_ok=True)

    world = GenesisWorld(
        scene_init_kwargs=dict(
            sim_options=gs.options.SimOptions(dt=1 / 15, substeps=10),
            rigid_options=gs.options.RigidOptions(box_box_detection=True),
        ),
        # No scene lights — only the HDRI provides illumination.
        lights={},
    )

    table = TableBuilder(
        world,
        name="table",
        pos=(0.0, 0.0, TABLE_SURFACE_Z),
        yaw=TABLE_YAW,
        texture_path=_pick_fixed_table_texture(),
        # Neutral dark legs so the table reads as a piece of furniture under
        # any HDRI; helps judging the HDRI's background + lighting without
        # the legs stealing attention.
        leg_color=(0.30, 0.30, 0.30, 1.0),
    )

    franka = FrankaFR3WithRobotiq2F85(
        world,
        name="franka",
        pos=ROBOT_POS,
        quat=(1, 0, 0, 0),
    )

    cam_pos, cam_lookat = CAMERA_PRESETS[camera]
    cam_node = CameraSensorNode(
        world,
        name="preview_cam",
        option=CameraSensorNodeOptions(
            backend="nyx",
            pos=cam_pos,
            quat=CameraSensorNodeOptions.quat_from_lookat(cam_pos, cam_lookat),
            res=res,
            fov=60,
            spp=spp,
            denoise=True,
        ),
        render_rgb=True,
    )

    env = WorldEnv(
        world,
        [table, franka, cam_node],
        render_mode="rgb_array",
    )
    return env, world, cam_node, env_map_dir


# ---------------------------------------------------------------------------
# Per-HDRI rendering
# ---------------------------------------------------------------------------


def _render_one(
    env,
    world,
    *,
    hdr_path: Path,
    preview_path: Path,
    rotation_deg: float = 0.0,
) -> None:
    """Render one HDRI preview to ``preview_path``."""
    from unienv_genesis.options.light_option import GenesisWorldEnvMapOption

    world.env_map = GenesisWorldEnvMapOption(
        hdr_path=str(hdr_path),
        hdr_rotation=float(rotation_deg),
    )
    # ``reload=True`` forces a full scene rebuild so the Nyx camera picks up
    # the new env map (it is read at sensor-creation time during reload()).
    env.reset(reload=True)
    rgb = env.render()
    if rgb is None:
        raise RuntimeError("env.render() returned None; Nyx camera produced no frame.")
    _save_rgb_png(rgb, preview_path)


def _discover_hdrs(env_map_dir: Path, pattern: Optional[str]) -> List[Path]:
    """Return sorted HDR files in ``env_map_dir`` matching the optional glob."""
    all_hdrs = sorted(env_map_dir.glob("*.hdr"))
    if pattern is None:
        return all_hdrs
    pat = pattern.lower()
    return [p for p in all_hdrs if fnmatch.fnmatch(p.name.lower(), pat)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Render one preview PNG per HDRI in "
            "unienv_genesis_collection/assets/scenes/env_maps, "
            "and maintain an info.json manifest next to the HDRs."
        ),
    )
    p.add_argument(
        "--env-map-dir",
        type=Path,
        default=_DEFAULT_ENV_MAP_DIR,
        help=(
            "Directory containing the .hdr files. "
            f"Default: {_DEFAULT_ENV_MAP_DIR}"
        ),
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Path to the curation manifest. "
            "Default: <env-map-dir>/info.json"
        ),
    )
    p.add_argument(
        "--previews-dir",
        type=Path,
        default=None,
        help=(
            "Directory for rendered preview PNGs. "
            "Default: <env-map-dir>/previews"
        ),
    )
    p.add_argument(
        "--pattern",
        type=str,
        default=None,
        help=(
            "Optional case-insensitive glob (matched against the .hdr "
            "filename) to limit which HDRs are rendered. "
            "Example: --pattern '*studio*'."
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Render at most N HDRs (after --pattern filtering).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-render even if a preview PNG already exists on disk.",
    )
    p.add_argument(
        "--camera",
        type=str,
        default="front",
        choices=sorted(CAMERA_PRESETS.keys()),
        help="Which camera preset to use (default: front).",
    )
    p.add_argument(
        "--spp",
        type=int,
        default=64,
        help=(
            "Samples per pixel for the Nyx path tracer. Lower = faster, "
            "noisier. Default 64 (good for previews)."
        ),
    )
    p.add_argument(
        "--res",
        type=_parse_resolution,
        default=(640, 360),
        help=(
            "Render resolution as WIDTHxHEIGHT. Default 640x360 (lightweight)."
        ),
    )
    p.add_argument(
        "--backend",
        type=str,
        default="gpu",
        choices=["cpu", "gpu", "metal"],
        help="Genesis compute backend (default: gpu).",
    )
    p.add_argument(
        "--precision",
        type=str,
        default="32",
        choices=["32", "64"],
        help="Genesis floating-point precision (default: 32).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be rendered without actually rendering anything.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-HDRI progress output (only the final summary).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)

    env_map_dir: Path = args.env_map_dir.resolve()
    previews_dir: Path = (args.previews_dir or env_map_dir / _DEFAULT_PREVIEW_SUBDIR).resolve()
    manifest_path: Path = (args.manifest or env_map_dir / _DEFAULT_MANIFEST).resolve()

    if not env_map_dir.is_dir():
        print(f"ERROR: env-map directory not found: {env_map_dir}", file=sys.stderr)
        print(
            "  Hint: run scripts/prepare_env_maps.py first to download HDRs.",
            file=sys.stderr,
        )
        return 2

    hdrs = _discover_hdrs(env_map_dir, args.pattern)
    if args.limit is not None:
        hdrs = hdrs[: args.limit]
    if not hdrs:
        print(
            f"No .hdr files matched in {env_map_dir} (pattern={args.pattern!r})."
        )
        return 1

    manifest = _load_manifest(manifest_path)
    manifest["_generated_at"] = _now_iso()
    if "_schema_version" not in manifest:
        manifest["_schema_version"] = _MANIFEST_SCHEMA_VERSION

    previews_dir.mkdir(parents=True, exist_ok=True)

    print(f"Env maps:    {env_map_dir}")
    print(f"Previews:    {previews_dir}")
    print(f"Manifest:    {manifest_path}")
    print(
        f"Pattern:     {args.pattern!r}    "
        f"Limit: {args.limit if args.limit is not None else 'none'}    "
        f"Force: {args.force}    Camera: {args.camera}    "
        f"SPP: {args.spp}    Res: {args.res[0]}x{args.res[1]}"
    )
    print(f"Found {len(hdrs)} HDRI(s) to consider.")

    if args.dry_run:
        for h in hdrs:
            print(f"  would render: {h.name}")
        return 0

    env, world, _, _ = _build_world(
        backend_name=args.backend,
        spp=args.spp,
        precision=args.precision,
        camera=args.camera,
        res=args.res,
    )

    rendered = 0
    skipped = 0
    failed = 0
    started = time.time()
    try:
        for idx, hdr_path in enumerate(hdrs, start=1):
            stem = hdr_path.stem
            preview_path = previews_dir / f"{stem}.png"
            hdr_relpath = hdr_path.name
            preview_relpath = str(preview_path.relative_to(env_map_dir))

            existing = manifest.get("entries", {}).get(stem)
            already_rendered = preview_path.is_file() and not args.force

            if already_rendered:
                # Don't waste time re-rendering; just refresh the manifest
                # entry's ``rendered_at``/``preview_path`` to reflect the
                # current state (e.g. if the user moved the preview folder).
                if existing is None or existing.get("preview_path") != preview_relpath:
                    entry = _build_entry(
                        stem=stem,
                        hdr_relpath=hdr_relpath,
                        preview_relpath=preview_relpath,
                        existing=existing,
                        rendered_at=existing.get("rendered_at") if existing else _now_iso(),
                    )
                    manifest.setdefault("entries", {})[stem] = entry
                skipped += 1
                if not args.quiet:
                    print(
                        f"  [{idx:>4}/{len(hdrs)}] skip  {hdr_path.name} "
                        f"(preview already exists)"
                    )
                continue

            if not args.quiet:
                print(f"  [{idx:>4}/{len(hdrs)}] render {hdr_path.name} ...", flush=True)
            try:
                rotation_deg = 0.0
                if isinstance(existing, dict):
                    try:
                        rotation_deg = float(existing.get("rotation_deg", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        rotation_deg = 0.0
                _render_one(
                    env,
                    world,
                    hdr_path=hdr_path,
                    preview_path=preview_path,
                    rotation_deg=rotation_deg,
                )
            except Exception as e:
                failed += 1
                if not args.quiet:
                    print(
                        f"      FAIL  {hdr_path.name}: {e}",
                        file=sys.stderr,
                    )
                    traceback.print_exc(file=sys.stderr)
                # Record the failure in the manifest so the user can spot it.
                entry = _build_entry(
                    stem=stem,
                    hdr_relpath=hdr_relpath,
                    preview_relpath=preview_relpath,
                    existing=existing,
                    rendered_at=_now_iso(),
                    status=_AUTO_FAILED_STATUS,
                    last_error=str(e),
                )
                manifest.setdefault("entries", {})[stem] = entry
                _save_manifest(manifest_path, manifest)
                continue

            entry = _build_entry(
                stem=stem,
                hdr_relpath=hdr_relpath,
                preview_relpath=preview_relpath,
                existing=existing,
                rendered_at=_now_iso(),
            )
            manifest.setdefault("entries", {})[stem] = entry
            rendered += 1
            # Periodically flush the manifest so a long run is never lost.
            _save_manifest(manifest_path, manifest)
    finally:
        env.close()

    elapsed = time.time() - started
    print()
    print(
        f"Done in {elapsed:.1f}s — "
        f"rendered {rendered}, skipped {skipped}, failed {failed} "
        f"(of {len(hdrs)} HDRI(s))."
    )
    print(f"Manifest written to: {manifest_path}")
    return 0 if failed == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
