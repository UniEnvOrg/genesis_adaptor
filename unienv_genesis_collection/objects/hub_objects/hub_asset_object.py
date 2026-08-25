"""Base class for mesh objects whose assets live on the Hugging Face Hub.

Dataset objects (YCB, Google Scanned Objects, ...) are multi-file assets: a
description file (URDF/MJCF) referencing a visual mesh, textures and several
collision meshes, all inside one per-object directory on the collection's
assets repo.  :class:`HubAssetMeshObject` provides the shared machinery:

* Offline-safe construction — local paths are resolved eagerly (against
  ``asset_path.ASSETS_DIR``), nothing is downloaded.
* Lazy, *verified* fetching — on first scene load or geometry query the
  object directory is downloaded if any required file is missing.  A plain
  directory-existence check is not enough: an interrupted download can
  leave the directory half-populated, which would render untextured or
  fail at parse time.  Subclasses enumerate the full set of required files
  (including the meshes referenced by their description file) via
  :meth:`_required_files`.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from unienv_genesis_collection import asset_path
from unienv_genesis_collection.objects.mesh_object import MeshObject

__all__ = ["HubAssetMeshObject", "xml_file_references"]


def xml_file_references(xml_file: Path, attribute: str = "file") -> List[str]:
    """Collect file references from an XML description file.

    Returns the value of *attribute* for every element that carries it —
    e.g. ``filename`` on URDF ``<mesh>`` elements, or ``file`` on MJCF
    ``<mesh>`` / ``<texture>`` assets.
    """
    root = ET.parse(xml_file).getroot()
    return [
        value
        for element in root.iter()
        if (value := element.get(attribute)) is not None
    ]


class HubAssetMeshObject(MeshObject):
    """A :class:`MeshObject` backed by a per-object directory on the Hub.

    Subclasses must implement :meth:`_required_files` and typically override
    :meth:`load_into_scene` to load their description file (URDF/MJCF)
    instead of the bare visual mesh.

    Args:
        asset_relative_dir: Directory of this object relative to the assets
            root, POSIX separators (e.g. ``"objects/ycb/banana"``).
        mesh_file_name: Name of the visual mesh inside the directory, used
            for trimesh-based geometry queries.
        scale: Forwarded to :class:`MeshObject`.
        name: Technical identifier.
        fixed: Whether the object is immovable.
        language_annotations: Optional English-language descriptions.
        visual_texture_file_name: Optional name of an image file inside
            the object directory (e.g. ``"texture.png"`` for GSO,
            ``"texture_map.png"`` for YCB) bound as the diffuse texture
            when the descriptor is loaded as a bare visual mesh.  Resolved
            against ``asset_dir`` and forwarded to
            :class:`MeshObject` as ``visual_texture_path``.  ``None``
            (the default) leaves the surface untextured.
    """

    def __init__(
        self,
        asset_relative_dir: str,
        mesh_file_name: str,
        scale: Union[float, Tuple[float, float, float]] = 1.0,
        name: str = "object",
        fixed: bool = False,
        language_annotations: Optional[Sequence[str]] = None,
        visual_texture_file_name: Optional[str] = None,
        standing_quaternion: Optional[Sequence[float]] = None,
    ):
        self.asset_relative_dir = asset_relative_dir
        # ASSETS_DIR is read via the module (and frozen here) so tests and
        # users can repoint it before constructing objects.
        self.asset_dir: Path = asset_path.ASSETS_DIR / asset_relative_dir
        visual_texture_path = (
            str(self.asset_dir / visual_texture_file_name)
            if visual_texture_file_name is not None
            else None
        )
        super().__init__(
            mesh_file=self.asset_dir / mesh_file_name,
            scale=scale,
            name=name,
            fixed=fixed,
            language_annotations=language_annotations,
            visual_texture_path=visual_texture_path,
            standing_quaternion=standing_quaternion,
        )
        self._assets_ready = False

    # ------------------------------------------------------------------
    # Asset resolution
    # ------------------------------------------------------------------

    def _required_files(self) -> List[Path]:
        """Files that must exist locally for the object to be usable.

        Should include every file the description file references (their
        number varies per object), which can only be fully evaluated once
        the description file itself is present.
        """
        raise NotImplementedError

    def _missing_files(self) -> List[Path]:
        return [p for p in self._required_files() if not p.is_file()]

    def _ensure_assets(self) -> None:
        """Download the object directory if any required file is missing."""
        if self._assets_ready:
            return
        if self._missing_files():
            asset_path.download_assets(
                allow_patterns=f"{self.asset_relative_dir}/**",
            )
            missing = self._missing_files()
            if missing:
                missing_names = ", ".join(p.name for p in missing)
                raise FileNotFoundError(
                    f"Asset directory '{self.asset_relative_dir}' is incomplete "
                    f"after download from {asset_path.REPO_ID} "
                    f"(missing: {missing_names}). Looked in {self.asset_dir}."
                )
        self._assets_ready = True

    def ensure_assets(self) -> None:
        """Public façade over :meth:`_ensure_assets`.

        Downstream consumers (e.g. managers that load this descriptor's raw
        visual mesh directly via ``gs.morphs.Mesh(file=desc.mesh_file, ...)``
        rather than via :meth:`load_into_scene`) need to ensure the Hub
        directory is materialised before the file path is read by the
        morph.  Calling this public helper keeps the manager decoupled from
        the private (and historically named) ``_ensure_assets`` underscore API.
        """
        self._ensure_assets()

    # ------------------------------------------------------------------
    # MeshObject overrides
    # ------------------------------------------------------------------

    def _load_trimesh(self):
        self._ensure_assets()
        return super()._load_trimesh()
