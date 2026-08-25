"""Catalog loading for Hub-hosted object datasets (YCB, GSO, ...).

Per-object metadata — language annotations, category tags and placement
hints — is not kept in the codebase.  Each dataset ships a
``catalog.json`` next to its assets on the Hugging Face repo (e.g.
``objects/ycb/catalog.json``), structured as::

    {
      "<object_id>": {
        "language_annotations": ["<broad>", ..., "<detailed>"],
        "tags": ["<tag>", ...],
        "standing_quaternion": [<w>, <x>, <y>, <z>]  // optional
      },
      ...
    }

Language annotations are ordered from broad to detailed (e.g. ``"ball"`` →
``"green ball"`` → ``"tennis ball"``) and are kept separate from the tags.
The optional ``standing_quaternion`` is the object's upright placement
orientation (world ``(w, x, y, z)``), consumed by
:meth:`ManipulatableObject.set_pos_3d` — e.g. a drill standing on its
battery pack with the bit facing forward.  Objects without the field use
the identity orientation, as before.

:func:`load_catalog` materializes the file through the standard asset
machinery (downloading it on first use — a single small file) and caches
the parsed result per resolved path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Optional, Tuple

from unienv_genesis_collection import asset_path

__all__ = ["ObjectInfo", "load_catalog"]


@dataclass(frozen=True)
class ObjectInfo:
    """Catalog metadata for one dataset object.

    Attributes:
        object_id: Directory name on the assets repo.
        language_annotations: Human-readable descriptions, ordered from
            broad to detailed.
        tags: Category tags for pool filtering, kept separate from the
            language annotations.  May be empty.
        standing_quaternion: Optional upright placement orientation
            ``(w, x, y, z)``; ``None`` (the common case) means the
            identity orientation.
    """

    object_id: str
    language_annotations: Tuple[str, ...]
    tags: FrozenSet[str]
    standing_quaternion: Optional[Tuple[float, float, float, float]] = None


# Parsed catalogs keyed by resolved local path.  Keying on the resolved path
# (rather than the relative one) keeps repointed ASSETS_DIRs — e.g. in tests
# — from seeing each other's entries.
_CACHE: Dict[Path, Dict[str, ObjectInfo]] = {}


def load_catalog(catalog_relative_path: str) -> Dict[str, ObjectInfo]:
    """Load (and cache) a dataset catalog from the assets folder.

    Args:
        catalog_relative_path: Path of the ``catalog.json`` relative to the
            assets root, e.g. ``"objects/ycb/catalog.json"``.  Downloaded
            from the Hub if not present locally.

    Returns:
        Mapping of object id to :class:`ObjectInfo`.
    """
    resolved = (asset_path.ASSETS_DIR / catalog_relative_path).resolve()
    if resolved not in _CACHE:
        local = asset_path.get_asset_path(catalog_relative_path)
        raw = json.loads(Path(local).read_text())
        _CACHE[resolved] = {
            object_id: ObjectInfo(
                object_id=object_id,
                language_annotations=tuple(entry["language_annotations"]),
                tags=frozenset(entry["tags"]),
                standing_quaternion=(
                    tuple(float(c) for c in entry["standing_quaternion"])
                    if entry.get("standing_quaternion") is not None
                    else None
                ),
            )
            for object_id, entry in raw.items()
        }
    return _CACHE[resolved]
