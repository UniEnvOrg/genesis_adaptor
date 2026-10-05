"""The packaged-tree asset-completeness check (was ``_verify_object_assets`` ×4).

One implementation; lightwheel's extensions are params/hooks:

* ``verify_object_assets(usd_file, asset_dir)`` — the three verbatim copies.
* ``verify_object_assets(..., allow_unresolvable_refs=True)`` — lightwheel's
  Locomotion/Apartment tolerance: a missing entry whose extension is neither
  a USD layer nor an image (the UE-export ``Foo.Foo`` material-record
  pattern, which exists nowhere in the bundle) is returned instead of
  raising; pxr tolerates those, so the gate's load check stays the tripwire.
* ``verify_object_assets(..., recover_texture=...)`` — lightwheel's upstream
  path-authoring repair as an injectable hook: a missing texture whose
  expected directory ships the same file under a stripped exporter prefix is
  copied under the referenced name (the
  :func:`recoverable_texture` hook resolves the shipped stand-in).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

_USD_SUFFIXES = (".usd", ".usda", ".usdc", ".usdz")
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".exr", ".hdr", ".bmp", ".webp", ".tif", ".tiff")

#: Public spellings (lightwheel's repackage_scene imports these).
USD_SUFFIXES = _USD_SUFFIXES
IMAGE_SUFFIXES = _IMAGE_SUFFIXES


def is_fatal_missing_ref(path: Path) -> bool:
    """A missing reference that must be fatal (a USD layer or a texture)."""
    suffix = path.suffix.lower()
    return suffix in _USD_SUFFIXES or suffix in _IMAGE_SUFFIXES


#: Backwards-compatible alias (lightwheel's pre-refactor private name).
_is_fatal_missing_ref = is_fatal_missing_ref


def recoverable_texture(path: Path, asset_dir: Path) -> Optional[Path]:
    """The shipped file a missing-but-prefixed texture reference resolves to.

    A *fatal* missing reference (USD layer / image) is repaired only when a
    sibling of the expected texture file carries the same (case-folded) name
    minus a leading exporter prefix — i.e. the referenced basename, folded,
    ends with the sibling's folded name and both keep the same suffix.  A USD
    layer or a texture without such a sibling is never "recovered" (None).
    """
    if path.suffix.lower() not in _IMAGE_SUFFIXES:
        return None
    expected_dir = path.parent
    folded = path.name.lower()
    if not expected_dir.is_dir():
        return None
    matches = [
        sibling
        for sibling in sorted(expected_dir.iterdir())
        if (
            sibling.is_file()
            and sibling.name.lower() != folded
            and sibling.suffix.lower() == path.suffix.lower()
            and folded.endswith(sibling.name.lower())
        )
    ]
    if len(matches) != 1:
        return None
    return matches[0]


#: Backwards-compatible alias (lightwheel's pre-refactor private name).
_recoverable_texture = recoverable_texture


def verify_object_assets(
    usd_file: Path,
    asset_dir: Path,
    *,
    allow_unresolvable_refs: bool = False,
    recover_texture: Optional[Callable[[Path, Path], Optional[Path]]] = None,
) -> List[Path]:
    """Every pxr layer-stack file + texture the stage references exists.

    Args:
        usd_file: The packaged root layer to check.
        asset_dir: The packaged per-object directory (the reference anchor).
        allow_unresolvable_refs: Tolerate (return) the non-USD/non-image
            missing entries instead of raising (the Locomotion/Apartment
            material-record pattern).
        recover_texture: Optional repair hook ``(missing_path, asset_dir) ->
            shipped stand-in | None``; a stand-in is copied under the
            referenced name into *asset_dir* (lightwheel's prefix-dropped
            texture repair) so the check can pass.

    Returns:
        The tolerated (non-USD/non-image) missing references — empty when
        *allow_unresolvable_refs* is ``False``.
    """
    from shutil import copy2

    from unienv_genesis_collection.objects.hub_objects.articulated.hub_articulated_object import (
        _usd_required_files,
    )

    missing = [p for p in _usd_required_files(usd_file, asset_dir) if not p.is_file()]
    unresolvable: List[Path] = []
    if allow_unresolvable_refs:
        fatal = [p for p in missing if is_fatal_missing_ref(p)]
        unresolvable = [p for p in missing if not is_fatal_missing_ref(p)]
        missing = fatal
    if recover_texture is not None and missing:
        for path in list(missing):
            stand_in = recover_texture(path, asset_dir)
            if stand_in is None:
                continue
            try:
                target = asset_dir / path.relative_to(asset_dir)
            except ValueError:
                continue  # outside the anchor: stays fatal
            target.parent.mkdir(parents=True, exist_ok=True)
            copy2(stand_in, target)
            missing.remove(path)
    if missing:
        names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(
            f"{usd_file.name} references files missing from the packaged tree: "
            f"{names} (checked {asset_dir})."
        )
    return unresolvable
