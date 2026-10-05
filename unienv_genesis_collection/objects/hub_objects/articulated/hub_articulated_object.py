"""Hub-backed base class for articulated furniture descriptors.

:class:`HubAssetArticulatedObject` is the Hub-backed counterpart of
:class:`~unienv_genesis_collection.objects.articulated_object.ArticulatedObject`
(plan ``§1.4``): the descriptor's assets are a *per-object directory* on the
collection's Hugging Face assets repo containing a description file (an MJCF
``model.xml`` for RoboCasa fixtures, a URDF for ReplicaCAD furniture, a USD
stage for ArtVIP furniture) plus
every mesh / texture it references.

It deliberately does **not** inherit
:class:`~unienv_genesis_collection.objects.hub_objects.hub_asset_object.HubAssetMeshObject`
— that class is built around a *single* ``mesh_file`` (``MeshObject``
geometry), while an articulated object has one visual mesh *per link*.  Only
the Hub *machinery* is reimplemented here (``asset_relative_dir`` /
``_required_files`` / ``_ensure_assets`` / ``ensure_assets``);
:func:`~unienv_genesis_collection.objects.hub_objects.hub_asset_object.xml_file_references`
is reused as-is for the reference enumeration.

The shared behaviour (see the module docstring of
``unienv_genesis_collection.objects.articulated_object``):

* **Offline-safe construction** — local paths are resolved eagerly (against
  ``asset_path.ASSETS_DIR``, read via the module so tests can repoint it),
  nothing is downloaded.
* **Lazy, *verified* fetching** — on first scene load or geometry query the
  object directory is downloaded if any required file is missing, and the
  download is verified for completeness (an interrupted download must not
  silently produce an untextured / unparseable asset).
* **Format-specific behaviour stays in the subclasses**: they implement
  :meth:`ArticulatedObject._load_link_meshes_at_reference` (offline FK for
  the reference-configuration union bbox, via
  :func:`~unienv_genesis_collection.objects.articulated_object.urdf_link_meshes_at_config`
  or :func:`...mjcf_link_meshes_at_config`) and
  :meth:`~unienv_genesis_collection.objects.base_object.ManipulatableObject.load_into_scene`
  (``gs.morphs.URDF`` vs ``gs.morphs.MJCF``).  This base implements neither.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Set, Tuple, Union

from unienv_genesis_collection import asset_path
from unienv_genesis_collection.objects.articulated_object import ArticulatedObject
from unienv_genesis_collection.objects.hub_objects.hub_asset_object import (
    xml_file_references,
)

__all__ = ["HubAssetArticulatedObject"]

# Image suffixes the USD texture enumeration checks (ArtVIP-style assets keep
# all textures under ``resource/img/**``; stray ``.mdl`` OEM materials are
# dropped by the packaging step and never enter the required set).
_USD_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".exr", ".hdr", ".bmp", ".webp", ".tif", ".tiff"}


def _udim_tiles(layer_dir: Path, asset_path: str) -> List[Path]:
    """Concrete texture tiles a ``<UDIM>`` placeholder resolves to.

    ``file.<UDIM>.png`` → the existing ``file.1001.png`` / ``file.1002.png`` /
    … tiles under *layer_dir*.  An empty list means no tile exists (the caller
    treats the placeholder itself as the required-but-missing path).
    """
    import re

    # Replace the <UDIM> placeholder with a 4-digit tile number glob.  Strip a
    # *single* leading "./" (USD authors ``./Textures/file.<UDIM>.png``); the
    # relative comparison below never carries it.  (``str.lstrip("./")`` would
    # also eat leading dots of the first real segment — e.g. ``.hidden/a.<UDIM>.png``
    # — so it must not be used here.)
    normalized = asset_path[2:] if asset_path.startswith("./") else asset_path
    stem_pattern = re.escape(normalized).replace(re.escape("<UDIM>"), "[0-9]" * 4)
    return [
        path
        for path in sorted(layer_dir.rglob("*"))
        if path.is_file() and re.fullmatch(stem_pattern, str(path.relative_to(layer_dir)))
    ]


def _sdf_layer_compositional_paths(layer_file: Path) -> Tuple[List[str], List[str]]:
    """The payload / reference asset paths authored in one USD layer.

    ``UsdUtils.ExtractExternalReferences`` reports sublayers, asset-valued
    attributes and *reference* arcs — a **payload** arc is not guaranteed to
    surface in its reference list (it does for some exporters/layer formats,
    not for others), so the layer's own ``payload`` fields are read directly
    at the Sdf level: every prim spec (including specs authored inside
    variant sets — a payload/reference inside a variant still composes when
    the variant is selected) contributes its authored ``Sdf.Payload.assetPath``
    / ``SdfReference.assetPath``.  Resolution is the resolver's: relative to
    the authoring layer's own directory.
    """
    from pxr import Sdf

    layer = Sdf.Layer.FindOrOpen(str(layer_file))
    if layer is None:
        return [], []
    payloads: List[str] = []
    references: List[str] = []

    def _walk(spec) -> None:
        for key, sink in (("payload", payloads), ("references", references)):
            if not spec.HasInfo(key):
                continue
            list_op = spec.GetInfo(key)
            if list_op is None:
                continue
            for item in list_op.GetAppliedItems():
                asset_path = getattr(item, "assetPath", None)
                if asset_path:
                    sink.append(asset_path)
        if hasattr(spec, "variantSets"):
            for variant_set in spec.variantSets.values():
                for variant_spec in variant_set.variantChildren.values():
                    _walk(variant_spec)
        for name in list(spec.nameChildren.keys()):
            _walk(spec.nameChildren[name])

    _walk(layer.pseudoRoot)
    return payloads, references


def _usd_required_files(
    usd_file: Path,
    asset_dir: Path,
    *,
    required_only: bool = False,
) -> List[Path]:
    """Every file a USD stage needs locally: the pxr sublayer/reference
    closure + the asset-valued attribute references (textures).

    Mirrors what a Usd stage open (the loader Genesis instantiates) will
    actually open:

    * ``UsdUtils.ExtractExternalReferences`` walked **recursively** over the
      sublayers/references is authoritative for the ``.usd`` payload side —
      relative paths resolve against each layer's own directory, exactly like
      the asset resolver does at load.  Payload arcs (which the extractor
      does not reliably surface) are additionally read per layer at the Sdf
      level (:func:`_sdf_layer_compositional_paths`) and recursed the same
      way — the nested payload layers of a scene stage (Lightwheel's
      ``tudi2/tudi2.usd``, ``snow/Road/*.usd``, …) carry their own texture
      references, so the closure is a fixpoint over every USD layer pulled
      in, not a one-level walk;
    * ``resource/img`` textures surface as asset-valued attributes of the
      material layers; a *missing* texture is required too (a partial
      download is repaired — a glob of the existing tree could not name a
      file that is not there);
    * stray ``.thumbs``/``*.mdl`` OEM materials are not required (the
      packaging step drops them — Genesis's usd-core loader never opens
      non-USD OEM materials);
    * close-coupled ``resource/`` image-suffix files that are *present*
      locally are additionally required (belt and braces for textures whose
      attribute type the extractor may skip).

    Args:
        usd_file: Path to the stage's root layer.
        asset_dir: The per-object asset directory.
        required_only: Skip asset-valued references that are neither USD
            layers nor images.  Those (UE-export ``Foo.Foo`` material
            records) exist nowhere in the bundle and never will — listing
            them makes ``_missing_files()`` permanently non-empty, so the
            download-on-demand path would re-fetch the object on every
            build and an offline completeness check would always fail.  The
            packager keeps the default (it warn-and-records them instead —
            see ``scripts/prepare_lightwheel_assets.py``); the descriptor's
            ``_required_files`` passes ``True``.

    Returns:
        ``[usd_file, *closure_files]`` (de-duplicated by the caller).
    """
    from pxr import UsdUtils

    files: List[Path] = []
    seen: Set[Path] = set()

    def _add(path: Path) -> None:
        if path not in seen:
            seen.add(path)
            files.append(path)

    def _require(resolved: Path) -> None:
        if required_only:
            suffix = resolved.suffix.lower()
            if (
                suffix not in (".usd", ".usda", ".usdc", ".usdz")
                and suffix not in _USD_IMAGE_SUFFIXES
            ):
                return  # never satisfiable (e.g. a UE material record) — skip
        _add(resolved)

    pending: List[Path] = [Path(usd_file).resolve()]
    while pending:
        layer_file = pending.pop()
        _add(layer_file)
        try:
            sublayers, assets, references = UsdUtils.ExtractExternalReferences(
                str(layer_file)
            )
        except Exception:  # noqa: BLE001 — the loader reports unreadable layers
            continue
        try:
            sdf_payloads, sdf_references = _sdf_layer_compositional_paths(layer_file)
        except Exception:  # noqa: BLE001 — an unreadable layer has no payloads
            sdf_payloads, sdf_references = [], []
        layer_paths = [
            *sublayers,
            *references,
            *sdf_references,
            *sdf_payloads,
        ]
        for layer_ref in layer_paths:
            resolved = (layer_file.parent / layer_ref).resolve()
            if resolved.suffix.lower() not in (".usd", ".usda", ".usdc", ".usdz"):
                continue
            pending.append(resolved)
        for asset_path in assets:
            if "<UDIM>" in asset_path:
                # A UDIM texture tile set is authored as a single asset-valued
                # attribute whose path carries the ``<UDIM>`` placeholder (the
                # USD convention: ``file.1001.png``, ``file.1002.png``, … are
                # the concrete tiles).  The loader resolves the placeholder to
                # the per-tile files at render time, so the required set is
                # the concrete tiles that exist — and a set with no concrete
                # tile at all is still required as the placeholder path (a
                # partial download surfaces it).
                resolved_tiles = _udim_tiles(layer_file.parent, asset_path)
                if resolved_tiles:
                    for tile in resolved_tiles:
                        _add(tile)
                else:
                    _require((layer_file.parent / asset_path).resolve())
                continue
            resolved = (layer_file.parent / asset_path).resolve()
            if resolved.suffix.lower() == ".mdl":
                continue  # OEM material dropped at packaging (never by usd-core)
            _require(resolved)
    resource_dir = asset_dir / "resource"
    if resource_dir.is_dir():
        for asset_file in sorted(resource_dir.rglob("*")):
            if asset_file.is_file() and asset_file.suffix.lower() in _USD_IMAGE_SUFFIXES:
                _add(asset_file)
    return files


def _mjcf_required_files(mjcf_file: Path) -> List[Path]:
    """Every file an MJCF references, plus everything pulled in via ``<include>``.

    Collection mirrors the loader that will actually open these files —
    ``genesis.utils.mjcf.build_model``:

    * ``<include>`` elements are resolved relative to the object directory
      (``build_model``'s ``parent_path`` starts empty for the main file) and,
      for includes nested inside an already-included file, accumulate on that
      parent include's *own path* — so the required set names exactly the
      paths ``build_model`` opens while it splices the tree recursively.
    * References to binary assets in an included file keep their semantics:
      ``<mesh file=…>`` is rewritten by ``build_model`` to be relative to the
      include's directory, while other file references (``<texture>``,
      ``<hfield>``, ``<skin>``, …) are not.
    * Directories come from the *main* file's ``<compiler>`` (an
      ``assetdir``/``meshdir``/``texturedir`` inside an included file is
      shadowed by the main one, which ``build_model`` absolutizes).

    Args:
        mjcf_file: Path to the main MJCF file.

    Returns:
        The MJCF file itself first, then every referenced file (meshes,
        textures, height fields, … and included MJCF files).  Files that
        cannot be parsed are returned without descending into them — the
        loader reports them; once they are downloaded the enumeration simply
        discovers the deeper references.
    """
    mjcf_file = Path(mjcf_file)
    object_dir = mjcf_file.parent
    root = ET.parse(mjcf_file).getroot()
    if root.tag != "mujoco":
        raise ValueError(f"Not an MJCF file (root tag {root.tag!r}): {mjcf_file}")
    compiler = root.find("compiler")

    def _base_dir(attr: Optional[str]) -> Path:
        if attr:
            directory = Path(attr)
            return directory if directory.is_absolute() else object_dir / directory
        return object_dir

    # Mirrors build_model: assetdir/meshdir/texturedir default to the main
    # file's directory and are absolutized against it.
    asset_dir = _base_dir(compiler.get("assetdir") if compiler is not None else None)
    mesh_dir = _base_dir(compiler.get("meshdir") if compiler is not None else None)
    texture_dir = _base_dir(compiler.get("texturedir") if compiler is not None else None)

    # (file, include-relative parent path) — mirrors build_model's
    # (xml_root, parent_path) work stack.
    pending: List[Tuple[Path, Optional[Path]]] = [(mjcf_file, None)]
    seen: Set[Path] = set()
    files: List[Path] = []
    while pending:
        xml_file, include_parent = pending.pop()
        if xml_file in seen:
            continue
        seen.add(xml_file)
        files.append(xml_file)
        try:
            xml_root = ET.parse(xml_file).getroot()
        except (ET.ParseError, OSError):
            continue  # the loader reports malformed / missing files
        for element in xml_root.iter():
            ref = element.get("file")
            if ref is None or element.tag == "include":
                continue
            if element.tag == "mesh":
                if include_parent is not None:
                    # build_model rewrites <mesh file=…> inside included
                    # files to be relative to the include's directory.
                    ref = str(include_parent.parent / ref)
                files.append(mesh_dir / ref)
            elif element.tag == "texture":
                files.append(texture_dir / ref)
            else:
                files.append(asset_dir / ref)
        # Direct children of the file's root, exactly like build_model.
        for include_element in xml_root.findall("include"):
            include_ref = include_element.get("file")
            if include_ref is None:
                continue
            rel = Path(include_ref) if include_parent is None else include_parent / include_ref
            # The include file is appended when it is processed off the
            # stack (before its own parse attempt), keeping a single entry
            # even when it is missing / unparseable.
            pending.append((object_dir / rel, rel))
    return files


class HubAssetArticulatedObject(ArticulatedObject):
    """An :class:`ArticulatedObject` backed by a per-object Hub directory.

    Subclasses implement
    :meth:`~unienv_genesis_collection.objects.articulated_object.ArticulatedObject._load_link_meshes_at_reference`
    (offline FK in the base-link frame for the union bbox:
    ``urdf_link_meshes_at_config`` for URDF,
    ``mjcf_link_meshes_at_config`` for MJCF,
    ``usd_link_meshes_at_config`` for USD),
    :meth:`~unienv_genesis_collection.objects.articulated_object.ArticulatedObject._load_link_local_meshes_at_reference`
    (the same FK with ``link_local=True`` — link-frame geometry for
    part-grasp enumeration) and
    :meth:`~unienv_genesis_collection.objects.base_object.ManipulatableObject.load_into_scene`
    (format-specific morph options).  This base provides only the Hub asset
    machinery and the referenced-file enumeration.

    Args:
        asset_relative_dir: Directory of this object relative to the assets
            root, POSIX separators (e.g. ``"objects/robocasa/cabinet_hinge"``).
        description_file_name: Name of the description file inside the
            directory (``"model.xml"`` for MJCF, ``"<id>.urdf"`` for URDF,
            ``"<id>.usd"`` for USD).
        name: Technical identifier used when loading into the scene.
        fixed: Whether the object is immovable.  Furniture defaults to
            ``True`` (placed once at load).
        reference_qpos: Joint name → position at the reference configuration
            used by the offline union bbox (FK keys only — not a registry).
        language_annotations: Optional English-language descriptions.
        scale: Uniform scale factor forwarded to the morphs by the concrete
            subclasses.  MJCF morphs only support uniform (scalar) scaling —
            enforced by the MJCF subclass; URDF also accepts per-axis
            ``(sx, sy, sz)`` tuples.
        standing_quaternion: Optional ``(w, x, y, z)`` upright placement
            orientation, forwarded to
            :class:`~unienv_genesis_collection.objects.base_object.ManipulatableObject`
            (catalog metadata may define one; randomised placement composes a
            random yaw on top of it).
    """

    def __init__(
        self,
        asset_relative_dir: str,
        description_file_name: str,
        name: str = "articulated_object",
        fixed: bool = True,
        reference_qpos: Optional[Mapping[str, float]] = None,
        language_annotations: Optional[Sequence[str]] = None,
        scale: Union[float, Tuple[float, float, float]] = 1.0,
        standing_quaternion: Optional[Sequence[float]] = None,
    ):
        self.asset_relative_dir = asset_relative_dir
        # ASSETS_DIR is read via the module (and frozen here) so tests and
        # users can repoint it before constructing objects.  Eager, hence
        # offline-safe: nothing is downloaded at construction.
        self.asset_dir: Path = asset_path.ASSETS_DIR / asset_relative_dir
        self.description_file: Path = self.asset_dir / description_file_name
        self.scale: Union[float, Tuple[float, float, float]] = scale
        self._assets_ready = False
        super().__init__(
            name=name,
            fixed=fixed,
            reference_qpos=reference_qpos,
            language_annotations=language_annotations,
        )
        # ``ArticulatedObject.__init__`` does not expose it yet; set it with the
        # exact same normalisation ``ManipulatableObject`` uses so catalog
        # metadata (optional ``standing_quaternion``) stays functional.
        self.standing_quaternion: Optional[Tuple[float, float, float, float]] = (
            tuple(float(c) for c in standing_quaternion)
            if standing_quaternion is not None
            else None
        )

    # ------------------------------------------------------------------
    # Asset resolution (reimplemented Hub machinery — see module docstring)
    # ------------------------------------------------------------------

    def _required_files(self) -> List[Path]:
        """The description file plus every local file it references.

        The description file's own presence gates the enumeration: until it
        is downloaded the (asset-dependent) set of referenced files cannot be
        known, so only the description file itself is required — mirroring
        ``HubAssetMeshObject`` / ``GSOObject``.  The format is detected from
        the file extension: ``.xml`` → MJCF (mesh/texture/height-field
        ``file=`` references, ``<include>`` files walked recursively),
        ``.urdf`` → URDF ``filename=`` references, ``.usd``/``.usda``/
        ``.usdc`` → USD (pxr layer stack + ``resource/img/**`` textures).
        """
        required: List[Path] = [self.description_file]
        if not self.description_file.is_file():
            return required
        suffix = self.description_file.suffix.lower()
        if suffix == ".xml":
            required += _mjcf_required_files(self.description_file)
        elif suffix == ".urdf":
            required += [
                self.asset_dir / ref
                for ref in xml_file_references(self.description_file, attribute="filename")
            ]
        elif suffix in (".usd", ".usda", ".usdc"):
            required += _usd_required_files(self.description_file, self.asset_dir)
        else:
            raise ValueError(
                f"Unsupported description file format {suffix!r} for "
                f"{self.name!r}: expected '.xml' (MJCF), '.urdf' or "
                f"'.usd'/'.usda'/'.usdc' ({self.description_file})."
            )
        # Several geoms/visuals may reference the same mesh file.
        return list(dict.fromkeys(required))

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

        Downstream consumers (e.g. a manager that loads this descriptor's
        description file directly) need the Hub directory materialised before
        the description file path is read by a morph.  Calling this public
        helper keeps the manager decoupled from the private (and historically
        named) ``_ensure_assets`` underscore API.
        """
        self._ensure_assets()
