#!/usr/bin/env python3
"""Download surface texture materials from Poly Haven (polyhaven.com/textures).

Poly Haven publishes a large collection of CC0 PBR texture materials.  This
script fetches the requested maps (by default just the diffuse / color map,
which is all that the current Genesis floor rendering needs) for a selection of
texture assets and saves them into per-asset subdirectories of a local folder
for use with floor / ground surfaces.

Usage::

    # Download 10 random floor textures (2k JPG diffuse) into the default dir.
    python scripts/prepare_floor_textures.py --count 10

    # Download specific textures by id (see https://polyhaven.com/textures).
    python scripts/prepare_floor_textures.py --ids aerial_asphalt_01 brick_wall_001

    # Filter by several floor-friendly categories and also grab normals.
    python scripts/prepare_floor_textures.py --categories floor tiles wood concrete \
        --maps diffuse nor_gl --count 20

    # Download *every* texture in the "floor" category at 4k.
    python scripts/prepare_floor_textures.py --categories floor --all --res 4k

The default output directory is the collection's asset tree so that floors can
reference the files via :func:`get_asset_path`.

API reference: https://api.polyhaven.com/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Poly Haven API endpoints.
_ASSETS_URL = "https://api.polyhaven.com/assets"
_FILES_URL = "https://api.polyhaven.com/files/{asset_id}"

# Asset type for textures (from the Poly Haven API).  0 == HDRI, 1 == texture,
# 2 == 3D model.
_TEXTURE_ASSET_TYPE = 1

# The Poly Haven CDN (dl.polyhaven.org) rejects requests with urllib's
# default User-Agent ("Python-urllib/...") with HTTP 403, so we send a
# conventional browser UA instead.
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# Default output directory: the collection's assets/scenes/floor_textures tree.
_DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent
    / "unienv_genesis_collection"
    / "assets"
    / "scenes"
    / "floor_textures"
)

# Floor-friendly categories used when the user does not pass --categories.
_DEFAULT_CATEGORIES: List[str] = ["floor"]

# Floor-related categories that are sensible to offer / document.
_FLOOR_RELATED_CATEGORIES = [
    "floor",
    "tiles",
    "wood",
    "concrete",
    "asphalt",
    "cobblestone",
    "road",
    "carpet",
    "indoor",
    "outdoor",
]

# Mapping from user-facing map names (lowercase) to the keys used by the
# Poly Haven files API.  Multiple aliases map to the same canonical key.
_MAP_ALIASES: Dict[str, str] = {
    "diffuse": "Diffuse",
    "diff": "Diffuse",
    "color": "Diffuse",
    "albedo": "Diffuse",
    "nor_gl": "nor_gl",
    "normal_gl": "nor_gl",
    "normal": "nor_gl",
    "nor_dx": "nor_dx",
    "normal_dx": "nor_dx",
    "rough": "Rough",
    "roughness": "Rough",
    "ao": "AO",
    "ambient_occlusion": "AO",
    "arm": "arm",
    "displacement": "Displacement",
    "disp": "Displacement",
    "rough_ao": "rough_ao",
}

# Default maps: just the diffuse / color map, which is what current floor
# rendering in Genesis consumes.
_DEFAULT_MAPS: List[str] = ["diffuse"]

PROVENANCE_README = """\
# Floor textures

Downloaded from [Poly Haven](https://polyhaven.com/textures) (CC0 / public
domain) by `genesis_adaptor/scripts/prepare_floor_textures.py`.

## Layout

Each texture material lives in its own subdirectory named after the Poly Haven
asset id, e.g.::

    floor_textures/
        aerial_asphalt_01/
            aerial_asphalt_01_diff_2k.jpg
        brick_wall_001/
            brick_wall_001_diff_2k.jpg
        ...

## Attribution

These textures are from [Poly Haven](https://polyhaven.com) — a collection of
100% free (CC0) textures. Please visit
[https://polyhaven.com/textures](https://polyhaven.com/textures) for the
original assets and consider
[supporting them](https://polyhaven.com/become-a-patron) on Patreon.

## Usage

Reference individual diffuse maps via the collection's asset path helper::

    from unienv_genesis_collection.asset_path import get_asset_path

    color_tex = get_asset_path(
        "scenes", "floor_textures", "aerial_asphalt_01", "aerial_asphalt_01_diff_2k.jpg",
    )
"""


def _urlopen(url: str, timeout: int = 300):
    """urlopen wrapper that sends a browser-like User-Agent."""
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310


def _fetch_json(url: str) -> object:
    """Fetch and parse a JSON document from *url*."""
    with _urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download(url: str, dest: Path, expected_md5: Optional[str] = None) -> None:
    """Download *url* to *dest*, verifying the MD5 checksum when provided."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    hasher = hashlib.md5()
    with _urlopen(url, timeout=300) as resp:
        data = resp.read()
    hasher.update(data)
    if expected_md5 is not None and hasher.hexdigest() != expected_md5:
        raise ValueError(
            f"MD5 mismatch for {dest.name}: expected {expected_md5}, "
            f"got {hasher.hexdigest()}"
        )
    dest.write_bytes(data)
    print(f"  saved {dest} ({len(data) / 1e6:.1f} MB)")


def _resolve_map_key(name: str) -> str:
    """Normalize a user-facing map name to its Poly Haven files-API key."""
    key = _MAP_ALIASES.get(name.lower())
    if key is None:
        raise ValueError(
            f"Unknown map '{name}'. Known maps: "
            f"{sorted(set(_MAP_ALIASES.keys()))}"
        )
    return key


def list_texture_ids(
    categories: Optional[Sequence[str]] = None,
    seed: Optional[int] = None,
    count: Optional[int] = None,
) -> List[str]:
    """Return a (possibly filtered / sampled) list of Poly Haven texture ids.

    Args:
        categories: If given, keep only assets whose ``categories`` list
            contains *any* of these strings (e.g. ``"floor"``, ``"tiles"``,
            ``"wood"``, ``"concrete"``).  ``None`` means no category filter.
        seed: Optional seed for deterministic random sampling.
        count: If given, randomly sample this many ids from the filtered set.

    Returns:
        Sorted list of texture asset ids.
    """
    import random

    assets = _fetch_json(_ASSETS_URL)
    cat_set = set(categories) if categories else None
    ids = [
        aid
        for aid, info in assets.items()
        if info.get("type") == _TEXTURE_ASSET_TYPE
        and (
            cat_set is None
            or cat_set.intersection(info.get("categories", []))
        )
    ]
    ids.sort()
    if count is not None and count < len(ids):
        rng = random.Random(seed)
        ids = rng.sample(ids, count)
        ids.sort()
    return ids


def _get_map_urls(
    asset_id: str, map_keys: Sequence[str], res: str, fmt: str
) -> List[tuple[str, str, str]]:
    """Return ``[(map_key, download_url, expected_md5), ...]`` for *asset_id*.

    Only maps that are actually available for *asset_id* at *res* in *fmt* are
    returned; missing maps are silently skipped.
    """
    files = _fetch_json(_FILES_URL.format(asset_id=asset_id))
    out: List[tuple[str, str, str]] = []
    for mk in map_keys:
        block = files.get(mk)
        if not block or res not in block:
            print(f"  {mk} not available at {res} for '{asset_id}' — skipping")
            continue
        entry = block[res].get(fmt)
        if entry is None:
            print(
                f"  {mk} has no '{fmt}' file at {res} for '{asset_id}' — skipping"
            )
            continue
        out.append((mk, entry["url"], entry.get("md5", "")))
    return out


def download_floor_textures(
    asset_ids: Sequence[str],
    output_dir: Path,
    maps: Sequence[str] = _DEFAULT_MAPS,
    res: str = "2k",
    fmt: str = "jpg",
) -> List[Path]:
    """Download the requested maps for each texture asset into per-asset dirs.

    Args:
        asset_ids: Poly Haven texture ids (e.g. ``"aerial_asphalt_01"``).
        output_dir: Destination directory (created if needed).  Each asset is
            written into its own subdirectory named after the id.
        maps: User-facing map names (e.g. ``"diffuse"``, ``"nor_gl"``).
        res: Resolution key (``"1k"``, ``"2k"``, ``"4k"``, ``"8k"``).
        fmt: Image format (``"jpg"``, ``"png"``, ``"exr"``).

    Returns:
        List of absolute :class:`Path` objects for the downloaded files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    map_keys = [_resolve_map_key(m) for m in maps]
    downloaded: List[Path] = []
    for aid in asset_ids:
        print(f"[{aid}]")
        dest_dir = output_dir / aid
        entries = _get_map_urls(aid, map_keys, res, fmt)
        if not entries:
            print(f"  no selected maps available for '{aid}' — skipping")
            continue
        for _mk, url, md5 in entries:
            dest = dest_dir / Path(url).name
            if dest.exists():
                print(f"  already present: {dest.name}  (skipping)")
            else:
                _download(url, dest, expected_md5=(md5 or None))
            downloaded.append(dest)
    return downloaded


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download Poly Haven texture materials for floors.",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="Specific Poly Haven texture ids to download. "
        "Overrides --count / --categories / --all.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of textures to (randomly) download when --ids / --all "
        "are not given. Default 5.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download every texture matching --categories (ignore --count).",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=list(_DEFAULT_CATEGORIES),
        help="Filter textures by Poly Haven category; a texture is kept if it "
        "matches ANY of the given categories. Floor-friendly options include: "
        f"{', '.join(_FLOOR_RELATED_CATEGORIES)}. Default: "
        f"{' '.join(_DEFAULT_CATEGORIES)}.",
    )
    parser.add_argument(
        "--maps",
        nargs="*",
        default=list(_DEFAULT_MAPS),
        help="Which PBR maps to download per asset. Aliases: "
        f"{', '.join(sorted(set(_MAP_ALIASES.keys())))}. "
        "Default: diffuse (current floor rendering only needs a color texture).",
    )
    parser.add_argument(
        "--res",
        type=str,
        default="2k",
        choices=["1k", "2k", "4k", "8k"],
        help="Resolution of the texture files to download. Default 2k.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="jpg",
        choices=["jpg", "png", "exr"],
        help="Image format to download. Default jpg (small, good for local use).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic sampling when using --count.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Destination directory for the downloaded textures. Each asset "
        "is written into its own subdirectory.",
    )
    parser.add_argument(
        "--write-readme",
        action="store_true",
        help="Write a provenance README.md into the output directory.",
    )
    args = parser.parse_args(argv)

    if args.ids is not None:
        if not args.ids:
            parser.error("--ids requires at least one id.")
        asset_ids = list(args.ids)
    else:
        cats = args.categories or None
        print(f"Listing textures (categories={cats!r}) ...")
        asset_ids = list_texture_ids(
            categories=cats,
            seed=args.seed,
            count=None if args.all else args.count,
        )

    if not asset_ids:
        print("No textures matched the criteria.")
        return 1

    print(
        f"Downloading {len(asset_ids)} texture(s) "
        f"(maps={args.maps}, res={args.res}, fmt={args.format}) "
        f"into {args.output_dir}"
    )
    paths = download_floor_textures(
        asset_ids,
        args.output_dir,
        maps=args.maps,
        res=args.res,
        fmt=args.format,
    )

    if args.write_readme:
        readme = args.output_dir / "README.md"
        readme.parent.mkdir(parents=True, exist_ok=True)
        readme.write_text(PROVENANCE_README)
        print(f"Wrote {readme}")

    print(f"\nDone. {len(paths)} file(s) in {args.output_dir}")
    print("Reference a floor color texture via:")
    print("  from unienv_genesis_collection.asset_path import get_asset_path")
    print("  tex = get_asset_path('scenes', 'floor_textures', '<asset_id>', '<file>.jpg')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
