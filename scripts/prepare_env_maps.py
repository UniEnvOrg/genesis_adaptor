#!/usr/bin/env python3
"""Download HDR environment maps from Poly Haven (polyhaven.com/hdris).

Poly Haven publishes a large collection of CC0 HDRIs.  This script fetches the
1k-resolution ``.hdr`` files (suitable for the Nyx renderer via
:class:`GenesisWorldEnvMapOption`) and saves them into a local directory for
use with :class:`EnvMapManager`.

Usage::

    # Download 10 random HDRIs (1k HDR format) into the default assets dir.
    python scripts/prepare_env_maps.py --count 10

    # Download specific HDRIs by id (see https://polyhaven.com/hdris).
    python scripts/prepare_env_maps.py --ids studio_small_09 abandoned_factory_canteen_01

    # Filter by category and use 2k resolution.
    python scripts/prepare_env_maps.py --category studio --res 2k --count 5 --output-dir /data/hdri

The default output directory is the collection's asset tree so that
``EnvMapManager(hdr_paths=...)`` can reference the files via
:func:`get_asset_path`.

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

# Asset type for HDRIs (from the Poly Haven API).
_HDRI_ASSET_TYPE = 0

# The Poly Haven CDN (dl.polyhaven.org) rejects requests with urllib's
# default User-Agent ("Python-urllib/...") with HTTP 403, so we send a
# conventional browser UA instead.
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


def _urlopen(url: str, timeout: int = 300):
    """urlopen wrapper that sends a browser-like User-Agent."""
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310

# Default output directory: the collection's assets/scenes/env_maps tree.
_DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent
    / "unienv_genesis_collection"
    / "assets"
    / "scenes"
    / "env_maps"
)

PROVENANCE_README = """\
# Environment maps (HDRIs)

Downloaded from [Poly Haven](https://polyhaven.com/hdris) (CC0 / public domain)
by `genesis_adaptor/scripts/prepare_env_maps.py`.

## Attribution

These HDRIs are from [Poly Haven](https://polyhaven.com) — a collection of
100% free (CC0) HDR textures. Please visit
[https://polyhaven.com/hdris](https://polyhaven.com/hdris) for the original
assets and consider [supporting them](https://polyhaven.com/become-a-patron)
on Patreon.

## Usage

Use with `EnvMapManager` — auto-discovery will pick up all `.hdr` files in
this directory automatically::

    from unienv_genesis_collection.scenes.env_map_manager import EnvMapManager

    manager = EnvMapManager(world)  # auto-discovers all .hdr files here

Or reference individual maps explicitly::

    from unienv_genesis_collection.asset_path import get_asset_path
    from unienv_genesis_collection.scenes.env_map_manager import EnvMapManager

    paths = [
        get_asset_path("scenes", "env_maps", "studio_small_09_1k.hdr"),
        ...
    ]
    manager = EnvMapManager(world, hdr_paths=paths)
"""


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


def list_hdri_ids(
    category: Optional[str] = None,
    seed: Optional[int] = None,
    count: Optional[int] = None,
) -> List[str]:
    """Return a (possibly filtered / sampled) list of Poly Haven HDRI ids.

    Args:
        category: If given, keep only assets whose ``categories`` list
            contains this string (e.g. ``"studio"``, ``"indoor"``,
            ``"outdoor"``, ``"skies"``).
        seed: Optional seed for deterministic random sampling.
        count: If given, randomly sample this many ids from the filtered set.

    Returns:
        Sorted list of HDRI asset ids.
    """
    import random

    assets = _fetch_json(_ASSETS_URL)
    ids = [
        aid
        for aid, info in assets.items()
        if info.get("type") == _HDRI_ASSET_TYPE
        and (category is None or category in info.get("categories", []))
    ]
    ids.sort()
    if count is not None and count < len(ids):
        rng = random.Random(seed)
        ids = rng.sample(ids, count)
        ids.sort()
    return ids


def _get_hdr_url(asset_id: str, res: str) -> tuple[str, str]:
    """Return ``(download_url, expected_md5)`` for the ``.hdr`` file of *asset_id*."""
    files = _fetch_json(_FILES_URL.format(asset_id=asset_id))
    hdri = files.get("hdri", {})
    if res not in hdri:
        raise KeyError(
            f"Resolution '{res}' not available for HDRI '{asset_id}'. "
            f"Available: {list(hdri.keys())}"
        )
    hdr = hdri[res].get("hdr")
    if hdr is None:
        raise KeyError(
            f".hdr format not available for '{asset_id}' at {res}. "
            f"Available formats: {list(hdri[res].keys())}"
        )
    return hdr["url"], hdr.get("md5", "")


def download_env_maps(
    asset_ids: Sequence[str],
    output_dir: Path,
    res: str = "1k",
) -> List[Path]:
    """Download the given HDRI asset ids as ``.hdr`` files into *output_dir*.

    Args:
        asset_ids: Poly Haven HDRI ids (e.g. ``"studio_small_09"``).
        output_dir: Destination directory (created if needed).
        res: Resolution key (``"1k"``, ``"2k"``, ``"4k"``, ``"8k"``).

    Returns:
        List of absolute :class:`Path` objects for the downloaded files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: List[Path] = []
    for aid in asset_ids:
        url, md5 = _get_hdr_url(aid, res)
        # The upstream filename already encodes id + resolution, e.g.
        # "studio_small_09_1k.hdr".
        dest = output_dir / Path(url).name
        if dest.exists():
            print(f"  already present: {dest.name}  (skipping)")
        else:
            _download(url, dest, expected_md5=(md5 or None))
        downloaded.append(dest)
    return downloaded


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download HDR env maps from Poly Haven.",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="Specific Poly Haven HDRI ids to download. "
        "Overrides --count / --category.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of HDRIs to (randomly) download when --ids is not given.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter HDRIs by Poly Haven category (e.g. studio, indoor, outdoor, skies).",
    )
    parser.add_argument(
        "--res",
        type=str,
        default="1k",
        choices=["1k", "2k", "4k", "8k"],
        help="Resolution of the .hdr file to download. Default 1k (~1-5 MB each).",
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
        help="Destination directory for the downloaded .hdr files.",
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
        print(f"Listing HDRIs (category={args.category!r}) ...")
        asset_ids = list_hdri_ids(
            category=args.category, seed=args.seed, count=args.count,
        )

    if not asset_ids:
        print("No HDRIs matched the criteria.")
        return 1

    print(f"Downloading {len(asset_ids)} HDRI(s) at {args.res} into {args.output_dir}")
    paths = download_env_maps(asset_ids, args.output_dir, res=args.res)

    if args.write_readme:
        readme = args.output_dir / "README.md"
        readme.write_text(PROVENANCE_README)
        print(f"Wrote {readme}")

    print(f"\nDone. {len(paths)} env map(s) in {args.output_dir}")
    print("Use with EnvMapManager:")
    print(f"  from pathlib import Path")
    print(f"  paths = [str(p) for p in sorted(Path({str(args.output_dir)!r}).glob('*.hdr'))]")
    print(f"  manager = EnvMapManager(world, hdr_paths=paths)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
