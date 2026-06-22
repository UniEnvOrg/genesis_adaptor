"""Asset path utilities for ``unienv_genesis_collection``.

Assets (meshes, URDF/XML, textures, ...) are *not* shipped inside the Python
package. They are hosted on the Hugging Face Hub at
``UniEnvOrg/unienv_genesis_collection_assets`` (a ``dataset`` repo) and
downloaded on demand into a local assets directory.

Typical usage::

    from unienv_genesis_collection.asset_path import get_asset_path

    # Lazily downloads just this file from the Hub if it is not present yet,
    # then returns its absolute local path.
    fr3 = get_asset_path("robots/franka_fr3/fr3.xml")

To pre-fetch everything (e.g. for offline use)::

    from unienv_genesis_collection.asset_path import download_assets
    download_assets()
"""

import os
from pathlib import Path
from typing import List, Optional, Union

import huggingface_hub

__all__ = [
    "REPO_ID",
    "REPO_TYPE",
    "ASSETS_REVISION",
    "ASSETS_DIR",
    "get_asset_path",
    "download_assets",
    "download_asset",
    "is_asset_available",
    "assets_present",
    "check_assets",
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Hugging Face Hub repository hosting the assets.
REPO_ID = "UniEnvOrg/unienv_genesis_collection_assets"
REPO_TYPE = "dataset"

# Revision (branch/tag/commit) to pull assets from.
ASSETS_REVISION = "main"

# Local directory where assets are materialized (next to this module).
ASSETS_DIR = Path(__file__).parent.resolve() / "assets"

def _hub_kwargs() -> dict:
    """Return Hub API kwargs using the current :data:`ASSETS_DIR`.

    ``ASSETS_DIR`` is occasionally overridden in tests or local sandbox setups.
    Build these kwargs at call time rather than import time so downloads always
    honor the current asset root.
    """
    return {
        "repo_id": REPO_ID,
        "repo_type": REPO_TYPE,
        "revision": ASSETS_REVISION,
        "local_dir": str(ASSETS_DIR),
    }


def _normalize_relative(*relative_paths: Union[str, os.PathLike]) -> str:
    """Join path components into a single POSIX-style relative path string.

    The Hub addresses files with ``/`` separators regardless of platform, so we
    normalize to forward slashes here.
    """
    if not relative_paths:
        raise ValueError("At least one path component is required.")

    rel = os.path.join(*[os.fspath(p) for p in relative_paths])
    return Path(rel).as_posix()


# ---------------------------------------------------------------------------
# Download utilities
# ---------------------------------------------------------------------------

def download_asset(*relative_paths: Union[str, os.PathLike], force: bool = False) -> Path:
    """Download a single asset file from the Hub into :data:`ASSETS_DIR`.

    Only the requested file is fetched (and cached), so this is cheap for
    on-demand access. Subsequent calls reuse the local copy unless ``force``.

    Args:
        *relative_paths: Path components relative to the assets root, e.g.
            ``"robots/franka_fr3/fr3.xml"`` or ``("robots", "franka_fr3", "fr3.xml")``.
        force: Re-download even if a local copy already exists.

    Returns:
        Absolute :class:`~pathlib.Path` to the downloaded file.
    """
    rel = _normalize_relative(*relative_paths)

    local_path = huggingface_hub.hf_hub_download(
        **_hub_kwargs(),
        filename=rel,
        force_download=force,
    )
    return Path(local_path).resolve()


def download_assets(
    *,
    allow_patterns: Optional[Union[str, List[str]]] = None,
    ignore_patterns: Optional[Union[str, List[str]]] = None,
    force: bool = False,
) -> Path:
    """Download the full asset collection (or a filtered subset) from the Hub.

    Useful for pre-fetching assets for offline use. Files already present are
    skipped unless ``force`` is set.

    Args:
        allow_patterns: Glob pattern(s) of files to include (e.g. ``"robots/**"``).
            If ``None``, all files are downloaded.
        ignore_patterns: Glob pattern(s) of files to exclude.
        force: Re-download files even if a local copy already exists.

    Returns:
        Absolute :class:`~pathlib.Path` to :data:`ASSETS_DIR`.
    """
    huggingface_hub.snapshot_download(
        **_hub_kwargs(),
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        force_download=force,
    )
    return ASSETS_DIR


# ---------------------------------------------------------------------------
# Checking utilities
# ---------------------------------------------------------------------------

def _list_remote_asset_files() -> List[str]:
    """Return the repository-relative paths of all asset files on the Hub.

    Repo-level metadata that lives at the repository root (``.gitattributes``,
    a ``README.md`` dataset card, ...) is excluded: every real asset lives in a
    subdirectory, so we keep only paths that contain a ``/``.
    """
    repo_files = huggingface_hub.list_repo_files(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        revision=ASSETS_REVISION,
    )
    return [f for f in repo_files if "/" in f]


def is_asset_available(*relative_paths: Union[str, os.PathLike]) -> bool:
    """Return ``True`` if the given asset already exists locally."""
    rel = _normalize_relative(*relative_paths)
    return (ASSETS_DIR / rel).is_file()


def assets_present() -> bool:
    """Return ``True`` if the local assets directory contains any asset files.

    This is a cheap, offline heuristic ("have assets been downloaded at all?"),
    not a completeness check. Use :func:`check_assets` to verify against the Hub.
    """
    if not ASSETS_DIR.is_dir():
        return False
    for path in ASSETS_DIR.rglob("*"):
        # Ignore huggingface_hub bookkeeping (e.g. the ``.cache`` directory).
        if path.is_file() and ".cache" not in path.relative_to(ASSETS_DIR).parts:
            return True
    return False


def check_assets(download_missing: bool = False) -> List[str]:
    """Compare local assets against the Hub manifest and report what is missing.

    Queries the Hub for the list of files in the repository and checks each one
    against :data:`ASSETS_DIR`.

    Args:
        download_missing: If ``True``, download any missing files before
            returning.

    Returns:
        A list of repository-relative paths that were missing locally (before
        any download triggered by ``download_missing``). An empty list means the
        local collection is complete.
    """
    missing = [f for f in _list_remote_asset_files() if not (ASSETS_DIR / f).is_file()]

    if download_missing:
        for rel in missing:
            download_asset(rel)

    return missing


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def get_asset_path(
    *relative_paths: Union[str, os.PathLike],
    download: bool = True,
) -> Path:
    """Get the absolute local path to an asset, downloading it if necessary.

    If the asset is not present locally, it is downloaded from the Hugging Face
    Hub. Both files and directories are supported: requesting a directory that
    is not present locally fetches its entire subtree.

    Args:
        *relative_paths: Path components relative to the assets directory.
            A single string like ``"robots/franka_fr3/fr3.xml"`` or multiple
            components like ``("robots", "franka_fr3", "fr3.xml")``.
        download: Whether to download the asset if it is missing (default
            ``True``). Set to ``False`` to require an already-present local copy.

    Returns:
        Absolute :class:`~pathlib.Path` to the asset file or directory.

    Raises:
        FileNotFoundError: If the asset does not exist locally and downloading is
            disabled, or if it cannot be found on the Hub.

    Example:
        >>> from unienv_genesis_collection.asset_path import get_asset_path
        >>> fr3_path = get_asset_path("robots/franka_fr3/fr3.xml")
        >>> fr3_path = get_asset_path("robots", "franka_fr3", "fr3.xml")
    """
    rel = _normalize_relative(*relative_paths)
    absolute_path = (ASSETS_DIR / rel).resolve()

    if absolute_path.exists():
        return absolute_path

    if not download:
        raise FileNotFoundError(
            f"Asset not found locally: {rel} (looked in {absolute_path}). "
            f"Downloading is disabled; call download_assets() or pass "
            f"download=True to fetch it from {REPO_ID}."
        )

    try:
        # The common case: a single file.
        return download_asset(rel)
    except Exception as file_exc:
        # The requested path may be a directory rather than a single file;
        # fall back to downloading its whole subtree.
        try:
            download_assets(allow_patterns=f"{rel}/**")
        except Exception:
            pass
        if absolute_path.exists():
            return absolute_path
        raise FileNotFoundError(
            f"Asset '{rel}' is not available locally and could not be "
            f"downloaded from {REPO_TYPE} repo '{REPO_ID}' (revision "
            f"'{ASSETS_REVISION}'): {file_exc}"
        ) from file_exc
