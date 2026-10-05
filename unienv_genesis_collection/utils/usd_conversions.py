"""Offline OpenUSD converters shared by the asset-prep scripts.

Both recipes run **offline** (usd-core only — no Genesis scene) during
packaging, rewriting a source stage into the layout the collection ships:

* :func:`swap_usd_joint_bodies` — MolmoSpaces authors every UsdPhysics joint
  with ``body0`` = **child** and ``body1`` = **parent** (the MJCF/PhysX
  export order), while Genesis's USD loader reads ``body0`` as the *parent*
  and ``body1`` as the child (``genesis/utils/usd/usd_rigid_entity.py``: a
  link is the joint's ``body1``).  Loading such a stage as-is either raises
  *Link … has multiple parents* or silently puts the movable part in the
  base link (the oven swings around the door).  The fix swaps the
  ``physics:body0`` ⇄ ``physics:body1`` relationship targets **and** the
  ``physics:localPos0/1`` + ``physics:localRot0/1`` attribute values, so every
  joint frame stays attached to the body it was authored for — verified 18/18
  samples single-entity with the correct base link and DOF kinds after the
  swap (``/tmp/cookie_agent_quantumcookie/molmospaces/swapped_all``).

* :func:`convert_mdl_to_preview_surface` — Omniverse/SimReady assets author
  their shading as **OmniPBR MDL** shaders (``info:implementationSource =
  'sourceAsset'``, ``info:mdl:sourceAsset = @OmniPBR.mdl@``), which usd-core
  cannot read: Genesis loses the materials *silently* (default grey
  surfaces).  The converter rewrites every MDL material graph into a native
  ``UsdPreviewSurface`` + ``UsdUVTexture`` node graph, leaves the articulation
  untouched, and strips the MDL shader prim so no non-USD material reference
  remains for the loader to see.

Both functions are stage edits (``Usd.Stage`` open + save), not text
rewrites, so the authored subtree survives regardless of the source's
ASCII/binary encoding.  All ``pxr`` imports stay lazy inside the functions.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional, Tuple

__all__ = [
    "joint_prim_of",
    "swap_joint_prim_bodies",
    "swap_usd_joint_bodies",
    "count_mdl_materials",
    "convert_mdl_to_preview_surface",
    "strip_unresolvable_physics_references",
    "hide_degenerate_mesh_prims",
]

logger = logging.getLogger(__name__)

# (side-0 attribute, side-1 attribute, sdf type) for each frame attribute
# authored *per side* of a joint: ``localPos0``/``localRot0`` live in the
# body0 target's frame, so when the body roles swap the values must swap
# attribute names with them.
_SIDE_ATTRS: Tuple[Tuple[str, str, str], ...] = (
    ("physics:localPos0", "physics:localPos1", "point3f"),
    ("physics:localPos1", "physics:localPos0", "point3f"),
    ("physics:localRot0", "physics:localRot1", "quatf"),
    ("physics:localRot1", "physics:localRot0", "quatf"),
)

_MDLSUF = ("point3f", "quatf")


def joint_prim_of(prim):
    """The UsdPhysics joint schema of a prim (``None`` when not a joint).

    Covers Revolute/Prismatic/Spherical/Fixed: a body0↔body1 inversion
    corrupts the welded link graph exactly like the moving DOFs.
    """
    from pxr import UsdPhysics

    for cls in (UsdPhysics.RevoluteJoint, UsdPhysics.PrismaticJoint,
                UsdPhysics.SphericalJoint, UsdPhysics.FixedJoint):
        if prim.IsA(cls):
            return cls(prim)
    return None


def swap_joint_prim_bodies(prim) -> bool:
    """Swap ``physics:body0`` ⇄ ``physics:body1`` on one joint prim.

    Mechanical and symmetrical:

    1. the ``physics:body0`` relationship targets become ``physics:body1``'s
       and vice versa;
    2. every authored ``physics:localPos0``/``localRot0`` value is re-authored
       as ``physics:localPos1``/``localRot1`` (and the other way round) — so
       each frame stays attached to the body it was authored for.

    Returns:
        ``True`` when *prim* is a UsdPhysics joint (whether or not it carried
        authored body relationships before).
    """
    from pxr import Sdf

    joint = joint_prim_of(prim)
    if joint is None:
        return False
    joint_prim = joint.GetPrim()

    body0_rel = joint_prim.GetRelationship("physics:body0")
    body1_rel = joint_prim.GetRelationship("physics:body1")
    if body0_rel is None and body1_rel is None:
        return True  # neither relationship authored: nothing to swap

    captured = {}
    for src, dst, type_name in _SIDE_ATTRS:
        attribute = joint_prim.GetAttribute(src)
        if attribute is not None and attribute.HasValue():
            captured[src] = attribute.Get()

    targets0 = list(body0_rel.GetTargets()) if body0_rel is not None else []
    targets1 = list(body1_rel.GetTargets()) if body1_rel is not None else []
    if body0_rel is not None:
        body0_rel.SetTargets(targets1)
    if body1_rel is not None:
        body1_rel.SetTargets(targets0)

    for src, dst, type_name in _SIDE_ATTRS:
        if src not in captured:
            continue  # this side's frame was not authored — leave it at default
        attribute = joint_prim.GetAttribute(dst)
        if attribute is None:
            attribute = joint_prim.CreateAttribute(dst, Sdf.ValueTypeNames.Find(type_name))
        attribute.Clear()  # drop the (opposite-side-authored) spec before setting
        attribute.Set(captured[src])
    return True


def swap_usd_joint_bodies(usd_file) -> int:
    """Swap body0 ⇄ body1 on every joint prim of the stage at *usd_file*.

    An in-place stage open + save keeps the composed subtree (mesh geometry,
    materials, collider variants) untouched — only joint prim specs change.
    MolmoSpaces per-object stages author everything (joints included) in the
    root layer, so the save is complete for them.

    Returns:
        The number of joint prims visited.
    """
    from pxr import Usd

    stage = Usd.Stage.Open(str(usd_file))
    visited = 0
    for prim in list(stage.Traverse()):  # bind prims before iterating
        if swap_joint_prim_bodies(prim):
            visited += 1
    stage.GetRootLayer().Save()
    return visited


# ---------------------------------------------------------------------------
# MDL (OmniPBR) → UsdPreviewSurface
# ---------------------------------------------------------------------------

_MDL_TEXTURE_SUFFIXES = (".png", ".jpg", ".jpeg", ".exr", ".hdr", ".tif", ".tiff", ".bmp")


def _texture_path(value) -> Optional[str]:
    """Normalize an authored asset attribute value to its plain path string.

    ``shader.GetInput("diffuse_texture").Get()`` returns an ``Sdf.AssetPath``;
    its ``str()`` renders with the ``@…@`` markers, so the raw ``.path`` is
    used instead (``str(value)``-only comparisons would never match a suffix).
    """
    if value is None:
        return None
    path = getattr(value, "path", value)
    if not isinstance(path, str):
        return None
    return path.strip("@ ")


def _texture_anchor(stage, attribute) -> Optional[Tuple[Path, str]]:
    """The ``(authoring_layer_dir, authored_path)`` of an asset-valued attribute.

    The MDL material graphs of a composed stage may live in a *nested payload
    layer* (Lightwheel's ``tudi2/tudi2.usd``, ``snow/Road/*.usd``, …), not in
    the root layer.  USD resolves an asset-valued attribute against the
    directory of the layer that **authors** the winning opinion, so the raw
    composed value cannot be re-anchored correctly without that layer: the
    converter re-authors every texture input into the root layer, where the
    same plain-relative path would resolve against the root's directory
    instead.  The winning site comes straight off the attribute's resolve
    info (its Pcp node: layer + authored path); an opinion authored in an
    anonymous (session) layer or a pure fallback has no on-disk anchor.
    """
    from pxr import Usd

    if attribute is None or not attribute.HasAuthoredValue():
        return None
    node = attribute.GetResolveInfo().GetNode()
    if node is None:
        return None
    # The winning Pcp site: layer stack + authored (pre-composition) path.
    authored_path = node.site.path
    layer = next(
        (l for l in node.layerStack.layers if not l.anonymous and l.realPath),
        None,
    )
    if layer is None or authored_path is None or str(authored_path) == "/":
        return None
    path = _texture_path(attribute.Get())
    if not path:
        return None
    return Path(layer.realPath).resolve().parent, path


def _relocate_texture(layer_dir: Path, texture_path: Optional[str], root_dir: Path) -> Optional[str]:
    """Re-anchor an authored texture path onto the packaged root layer.

    *layer_dir* is the directory of the layer authoring the texture input
    (the directory USD resolves *texture_path* against); the converter
    re-authors the texture into the root layer, so the returned path must
    resolve against *root_dir*.  A texture that resolves outside *root_dir*
    raises :class:`ValueError` (nothing the pack copies could satisfy it).

    A plain relative path authored by the root layer itself is returned
    verbatim (the root layer's directory *is* the anchor — byte-identical
    to the pre-existing behaviour for root-layer materials).

    Two authored-path shapes cannot be resolved the generic way and get
    their own rewrite:

    * a **Windows drive-letter absolute** path (``E:/360MoveData/…``,
    ElectricKettle007/Sink004's exporter): POSIX treats ``E:/…`` as a
    *relative* name (``PurePosixPath("E:/…").is_absolute()`` is False), so
    the generic relocation would ship the drive path verbatim and the pack
    would miss the texture forever.  The exporter's tree keeps the basename
    in the asset's texture sidecar directory, so the packaged copy is
    located by basename and the input is rewritten to it (unique basename
    match; with several, only one sitting in a ``texture``-named directory
    is used).  Without a unique packaged match the path is returned
    unchanged — the packager's required-file check then flags it.
    * a **UDIM** template (``T_…_Albedo.<UDIM>.png`` — the NVIDIA
    PhysicalAI-SimReady Warehouse racks): usd-core does not expand
    ``<UDIM>`` (``Ar.OpenAsset`` returns None and Genesis's texture reader
    raises), so the input is rewritten to the *concrete first tile* that
    exists beside it (``.1001.png``).  Caveat, accepted: the props' UVs may
    span several tiles, so a single-tile pick is load-correct but
    texture-lossy.
    """
    if not texture_path:
        return texture_path
    from pathlib import PurePosixPath

    authored = PurePosixPath(texture_path)
    if _WINDOWS_DRIVE_LETTER.match(texture_path):
        return _relocate_windows_texture(texture_path, root_dir)
    resolved = Path(texture_path)
    if not resolved.is_absolute():
        resolved = layer_dir / resolved
    resolved = resolved.resolve()
    try:
        relative = resolved.relative_to(root_dir)
    except ValueError:
        raise ValueError(
            f"texture {texture_path!r} (authored by {layer_dir}) resolves to "
            f"{resolved}, outside the packaged root layer directory {root_dir}.  "
            "The MDL→UsdPreviewSurface rewrite cannot relocate it into the "
            "packaged tree — fix the upstream material or package the "
            "referenced directory."
        ) from None
    # EXR maps are unreadable by the loader's PIL-based texture reader
    # (``genesis/utils/usd/usd_material.py``); the packager transcodes them to
    # a sibling ``.png`` — point the converter-authored input at it.
    if resolved.suffix.lower() == ".exr":
        png = resolved.with_suffix(".png")
        if png.is_file():
            relative = png.relative_to(root_dir)
            return f"./{relative.as_posix()}"
    # UDIM templates are never expanded by usd-core: point the authored input
    # at the concrete first tile that actually exists beside the template.
    if _UDIM_TOKEN in resolved.name:
        tile = _first_udim_tile(resolved)
        if tile is not None:
            return f"./{tile.relative_to(root_dir).as_posix()}"
    if layer_dir == root_dir and not authored.is_absolute() and ".." not in authored.parts:
        return texture_path  # the root layer itself authors it: verbatim
    return f"./{relative.as_posix()}"


_WINDOWS_DRIVE_LETTER = re.compile(r"^[A-Za-z]:[/\\]")
_UDIM_TOKEN = "<UDIM>"
# Standard UDIM tile numbering (Mari/Maya): 1001..1010 per row, grids run to
# 1001..1100 — scanning the full grid keeps a 1001-absent tile-set usable.
_UDIM_TILE_RANGE = range(1001, 1101)


def _relocate_windows_texture(texture_path: str, root_dir: Path) -> str:
    """Rewrite a ``<drive>:/…`` texture path onto the packaged tree by basename."""
    posix_form = texture_path.replace("\\", "/")
    basename = posix_form.rsplit("/", 1)[-1]
    matches = [
        packaged for packaged in root_dir.rglob("*")
        if packaged.is_file() and packaged.name == basename
    ]
    if len(matches) > 1:
        # Disambiguate with the exporter's directory convention (the basename
        # sits in a texture sidecar dir both upstream and in the pack).
        parent_name = posix_form.rsplit("/", 2)[-2].lower()
        in_sidecar = [
            packaged for packaged in matches
            if packaged.parent.name.lower() == parent_name
            and parent_name in ("texture", "textures")
        ]
        if len(in_sidecar) == 1:
            matches = in_sidecar
    if len(matches) != 1:
        logger.warning(
            "Windows texture path %r has no unique packaged copy of %r under "
            "%s (%d match(es)) — rewriting it is impossible; the required-file "
            "check will flag it.",
            texture_path, basename, root_dir, len(matches),
        )
        return texture_path
    return f"./{matches[0].relative_to(root_dir).as_posix()}"


def _first_udim_tile(resolved: Path) -> Optional[Path]:
    """The numerically first existing ``…<UDIM>…`` sibling tile of *resolved*."""
    stem, marker, tail = resolved.name.partition(_UDIM_TOKEN)
    if not marker:
        return None
    for tile_number in _UDIM_TILE_RANGE:
        candidate = resolved.with_name(f"{stem}{tile_number}{tail}")
        if candidate.is_file():
            return candidate
    return None


def _endswith_texture(path) -> bool:
    path = _texture_path(path)
    return bool(path) and path.lower().endswith(_MDL_TEXTURE_SUFFIXES)


def _first_authored(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _slot(material_prim, name: str):
    """Child prim path under the material prim for a converter-authored shader."""
    return material_prim.GetPath().AppendChild(name)


def convert_mdl_to_preview_surface(usd_file, out_file=None, *,
                                   drop_missing_normal_maps: bool = False) -> int:
    """Rewrite every OmniPBR MDL material graph into ``UsdPreviewSurface``.

    The stage's root layer is exported to *out_file* (default: overwritten in
    place) and re-opened, so the rewrite never touches the source snapshot on
    disk (export-then-edit).

    Mapping — verified against the Lightwheel SimReady bundle and the NVIDIA
    PhysicalAI-SimReady-Warehouse-01 dataset (both author OmniPBR graphs
    bound to ``outputs:mdl:surface``; the generic ``outputs:surface`` is left
    unconnected upstream):

    ============================  =========================================
    OmniPBR MDL input             UsdPreviewSurface input
    ============================  =========================================
    ``diffuse_texture``           ``diffuseColor`` (rgb, sRGB source)
    ``diffuse_color_constant``    ``diffuseColor`` constant fallback
    ``ORM_texture``               one file, three readers — the g/b
                                  channels drive ``roughness``/``metallic``
                                  (packed occlusion/roughness/metallic)
    ``normalmap_texture``         ``normal`` (rgb, linear)
    ``reflectionroughness_``      ``roughness`` (r, linear) fallback when
    ``texture``                   no packed ORM map exists
    ``enable_opacity`` ∘          ``opacity`` constant
    ``opacity_constant``
    ============================  =========================================

    The MDL shader prim is removed after each graph is rewritten (its
    ``@…mdl@`` asset reference would otherwise dangle for usd-core, which
    cannot open ``.mdl``).  Materials already carrying a preview surface are
    left untouched.

    Texture paths: the converter re-authors every texture input into the
    root layer, so each authored path is resolved against the layer that
    authors the material (which may be a *nested payload layer* of the
    composed stage) and rewritten relative to the root layer's directory;
    one that resolves outside the root layer's directory raises
    :class:`ValueError` (nothing the pack copies could satisfy it) — see
    :func:`_texture_anchor` / :func:`_relocate_texture` (which also rewrites
    Windows drive-letter paths and ``<UDIM>`` templates onto the pack).

    With ``drop_missing_normal_maps`` (the Lightwheel repackage opts in) a
    normal-map input whose packaged target file does not exist drops the
    normal slot instead of being authored — Stove070's 11 dangling
    ``…_N001.png`` inputs.

    Returns:
        The number of converted material graphs.
    """
    from pxr import Sdf, Usd, UsdShade

    out_file = str(out_file) if out_file is not None else str(usd_file)
    if str(usd_file) == out_file:
        # In-place rewrite: exporting over a *live* crate layer corrupts it,
        # so the composed root layer is staged through a temporary ASCII
        # file and re-exported to the final destination.
        temp_file = Path(str(usd_file)).with_suffix(".converted.usda")
        stage = Usd.Stage.Open(str(usd_file))
        stage.GetRootLayer().Export(str(temp_file))
        stage = Usd.Stage.Open(str(temp_file))
        stage.GetRootLayer().Export(out_file)
        temp_file.unlink(missing_ok=True)
    else:
        stage = Usd.Stage.Open(str(usd_file))
        stage.GetRootLayer().Export(out_file)
    stage = Usd.Stage.Open(out_file)
    # Every texture input is re-authored into the root layer: each authored
    # path is resolved against its authoring layer and rewritten relative to
    # the root layer's directory (see :func:`_relocate_texture`).
    root_dir = Path(out_file).resolve().parent

    converted = 0
    # Bind the material prim list before iterating: RemovePrim below invalidates
    # the prims of a live traversal (pxr prims are weak proxies of the stage).
    for material in [UsdShade.Material(p) for p in stage.Traverse() if p.IsA(UsdShade.Material)]:
        material_prim = material.GetPrim()
        mdl_shader = _mdl_shader_of(material_prim)
        if mdl_shader is None:
            continue  # already UsdPreviewSurface (or no shading): leave it

        def _mdl_input(name):
            shader_input = mdl_shader.GetInput(name)
            return shader_input.Get() if shader_input is not None else None

        def _relocated(name):
            """The input's texture path re-anchored onto the root layer."""
            attribute = mdl_shader.GetPrim().GetAttribute(f"inputs:{name}")
            anchor = _texture_anchor(stage, attribute)
            if anchor is None:
                return None
            return _relocate_texture(anchor[0], anchor[1], root_dir)

        diffuse_texture = _relocated("diffuse_texture")
        diffuse_constant = _mdl_input("diffuse_color_constant")
        orm_texture = _relocated("ORM_texture")
        normal_texture = (
            _relocated("normalmap_texture")
            if _mdl_input("normalmap_texture") is not None
            else _relocated("normal_texture")
        )
        if (
            normal_texture is not None and _endswith_texture(normal_texture)
            and drop_missing_normal_maps and not (root_dir / normal_texture).is_file()
        ):
            # A *dangling* normal map (Stove070's 11 ``…_N001.png`` inputs —
            # the texture genuinely does not exist upstream): drop the slot
            # instead of authoring a missing reference that would fail the
            # packager's required-file check and drop the whole object.  A
            # normal map is cosmetic; the diffuse/ORM maps keep the object's
            # real colour.  Opt-in (the Lightwheel repackage): the converter's
            # default contract maps every authored slot regardless of whether
            # the referenced file ships.
            logger.info(
                "%s: dropping dangling normal-map texture input %r — %s does "
                "not exist in the packaged tree.",
                out_file, normal_texture, root_dir / normal_texture,
            )
            normal_texture = None
        roughness_texture = (
            _relocated("reflectionroughness_texture")
            if _mdl_input("reflectionroughness_texture") is not None
            else _relocated("reflection_roughness_texture")
        )
        enable_opacity = _mdl_input("enable_opacity")
        opacity_constant = _mdl_input("opacity_constant")

        surface_shader = _define_preview_surface(
            stage, material_prim, diffuse_constant, enable_opacity, opacity_constant,
        )
        if _endswith_texture(diffuse_texture):
            tex = _define_uv_texture(stage, material_prim, "diffuse", diffuse_texture)
            surface_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                tex.ConnectableAPI(), "rgb",
            )
        if orm_texture is not None and _endswith_texture(orm_texture):
            # One packed ORM texture: roughness from the g channel, metallic
            # from b (the r channel is occlusion — no PreviewSurface slot).
            for channel, preview_input in (("g", "roughness"), ("b", "metallic")):
                tex = _define_uv_texture(
                    stage, material_prim, f"orm_{channel}", orm_texture, channel,
                )
                surface_shader.CreateInput(
                    preview_input, Sdf.ValueTypeNames.Float,
                ).ConnectToSource(tex.ConnectableAPI(), "r")
        elif _endswith_texture(roughness_texture):
            tex = _define_uv_texture(stage, material_prim, "roughness", roughness_texture, "r")
            surface_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
                tex.ConnectableAPI(), "r",
            )
        if normal_texture is not None and _endswith_texture(normal_texture):
            tex = _define_uv_texture(stage, material_prim, "normal", normal_texture)
            surface_shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
                tex.ConnectableAPI(), "rgb",
            )

        material.CreateSurfaceOutput().ConnectToSource(surface_shader.ConnectableAPI(), "surface")
        # Drop the MDL shader prim and the material outputs that pointed at it
        # (dangling connections to a removed prim make usd-core raise
        # UsdExpiredPrimAccessError the moment anything reads the outputs).
        for attribute in list(material_prim.GetAttributes()):
            if attribute.GetName().startswith("outputs:mdl:"):
                material_prim.RemoveProperty(attribute.GetName())
        stage.RemovePrim(mdl_shader.GetPath())
        converted += 1

    stage.GetRootLayer().Save()
    return converted


def _mdl_shader_of(material_prim):
    """The MDL shader prim of a material (``None`` when it is already Usd)."""
    from pxr import Usd, UsdShade

    for prim in Usd.PrimRange(material_prim):
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        if shader.GetImplementationSourceAttr().Get() == "sourceAsset":
            return shader  # MDL (OmniPBR) — usd-core cannot read this
    return None


def _define_preview_surface(stage, material_prim, diffuse_constant, enable_opacity, opacity_constant):
    """Author the ``UsdPreviewSurface`` shader prim of one material."""
    from pxr import Gf, Sdf, UsdShade

    surface_path = _slot(material_prim, "UsdPreviewSurface")
    surface_shader = UsdShade.Shader.Define(stage, surface_path)
    surface_shader.CreateIdAttr("UsdPreviewSurface")
    if diffuse_constant is None:
        diffuse = (0.8, 0.8, 0.8)
    else:
        diffuse = tuple(float(channel) for channel in tuple(diffuse_constant)[:3])
    surface_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse))
    surface_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    surface_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    if enable_opacity and opacity_constant not in (None, 1.0):
        surface_shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity_constant))
    return surface_shader


def _define_uv_texture(stage, material_prim, label: str, texture_file, channel: Optional[str] = None):
    """Author ``UsdUVTexture`` (+ its ``st`` primvar reader) for one slot.

    With *channel* the texture exposes that single float output (``r``/``g``/
    ``b``); without one it authors an ``rgb`` float3 output (albedo/normal).
    """
    from pxr import Sdf, UsdShade

    tex_path = _slot(material_prim, f"Tex_{label}")
    tex = UsdShade.Shader.Define(stage, tex_path)
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture_file)))
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    reader = UsdShade.Shader.Define(stage, tex_path.AppendChild("stReader"))
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    if channel is not None:
        tex.CreateOutput(channel, Sdf.ValueTypeNames.Float)
    else:
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    return tex


def count_mdl_materials(usd_file) -> int:
    """Number of material prims whose surface shader is an MDL shader."""
    from pxr import Usd, UsdShade

    stage = Usd.Stage.Open(str(usd_file))
    return sum(
        1
        for prim in stage.Traverse()
        if prim.IsA(UsdShade.Material) and _mdl_shader_of(prim) is not None
    )


# A convex hull needs a 3-D simplex: qhull raises ``QH6214 not enough points
# (3) to construct initial simplex (need 4)`` below four points.
_CONVEX_HULL_MIN_POINTS = 4


def hide_degenerate_mesh_prims(usd_file, out_file=None) -> int:
    """Make every mesh prim that cannot form a convex hull invisible.

    The PhysicalAI-SimReady Warehouse props ship a few geometry prims with a
    bare ``points`` array of 3 vertices (``Plastic_Rough_Black_A`` sticker
    planes on the convertible hand trucks and inside the m06 composition's
    payload).  Genesis builds collision geometry for *every* drawable prim of
    a link — with the loader's default ``^[vV]isual.*``/``^[cC]ollision.*``
    name patterns a non-matching prim name is BOTH visual and collision
    (``UsdPhysics.CollisionAPI`` never gets consulted), so the 3-point mesh
    reaches qhull and kills the load with ``QhullError: QH6214``.

    Hiding the prim (``UsdGeom.Imageable.MakeInvisible()``) removes it from
    Genesis's geometry walk entirely (``parse_prim_geoms`` skips
    ``ComputeVisibility() == invisible`` prims for the visual AND the
    collision role) while keeping the prim in the stage — no re-pathing, no
    destroyed indices.  Composed *instance proxies* are read-only and are
    skipped: for the warehouse assemblies the rewrite is applied to each
    closure prop stage file instead, where the mesh is a plain editable prim.

    The rewrite is export-then-edit (like
    :func:`convert_mdl_to_preview_surface`) so the source snapshot is never
    touched when *out_file* equals *usd_file*.

    Returns:
        The number of prims hidden.
    """
    from pxr import Usd, UsdGeom

    out_file = str(out_file) if out_file is not None else str(usd_file)
    if str(usd_file) == out_file:
        temp_file = Path(str(usd_file)).with_suffix(".degen-hide.usda")
        stage = Usd.Stage.Open(str(usd_file))
        stage.GetRootLayer().Export(str(temp_file))
        stage = Usd.Stage.Open(str(temp_file))
        stage.GetRootLayer().Export(out_file)
        temp_file.unlink(missing_ok=True)
    else:
        stage = Usd.Stage.Open(str(usd_file))
        stage.GetRootLayer().Export(out_file)
    stage = Usd.Stage.Open(out_file)

    to_hide: list = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or prim.IsInstanceProxy():
            continue
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if points is None or len(points) >= _CONVEX_HULL_MIN_POINTS:
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == "invisible":
            continue  # already hidden (an earlier pass) — idempotent
        to_hide.append(prim)
    for prim in to_hide:
        logger.info(
            "%s: hiding degenerate %d-point mesh prim %s (a convex hull is "
            "impossible and the loader's name-pattern fallback makes it a "
            "collision geom).",
            out_file, len(UsdGeom.Mesh(prim).GetPointsAttr().Get()), prim.GetPath(),
        )
        UsdGeom.Imageable(prim).MakeInvisible()

    if to_hide:
        stage.GetRootLayer().Save()
    return len(to_hide)


def strip_unresolvable_physics_references(usd_file, out_file=None) -> int:
    """Remove prims that reference unresolvable upstream physics layers.

    The NVIDIA PhysicalAI-SimReady Warehouse and similar SimReady bundles
    author their physics material (friction/restitution) as a prim under
    ``<root>/Materials/physics_<kind>`` carrying an **internal reference** to
    a shared layer addressed by an absolute ``omniverse://…`` URL (Omniverse
    Nucleus only).  That URL can never resolve on a plain filesystem, so the
    reference dangles and any asset-completeness check flags the packaged
    tree as incomplete — even though the packaging pipeline re-authors its own
    rigid-body + convex colliders and Genesis re-derives the physics material,
    making the upstream reference dead weight.

    This walks the prims, finds every one whose reference list contains an
    ``omniverse:``/absolute-URL asset, and removes the prim spec from its root
    layer.  Only references that cannot resolve to a file beside *usd_file*
    are touched (a resolvable relative reference is left alone).  Returns the
    number of prims removed.

    The rewrite is export-then-edit (like
    :func:`convert_mdl_to_preview_surface`) so the source snapshot is never
    touched when *out_file* equals *usd_file*.
    """
    from pxr import Sdf, Usd

    out_file = str(out_file) if out_file is not None else str(usd_file)
    if str(usd_file) == out_file:
        temp_file = Path(str(usd_file)).with_suffix(".pstrip.usda")
        stage = Usd.Stage.Open(str(usd_file))
        stage.GetRootLayer().Export(str(temp_file))
        stage = Usd.Stage.Open(str(temp_file))
        stage.GetRootLayer().Export(out_file)
        temp_file.unlink(missing_ok=True)
    else:
        stage = Usd.Stage.Open(str(usd_file))
        stage.GetRootLayer().Export(out_file)
    stage = Usd.Stage.Open(out_file)
    root_layer = stage.GetRootLayer()
    layer_dir = Path(out_file).parent

    def _unresolvable_reference(prim) -> bool:
        references = prim.GetMetadata("references")
        if references is None:
            return False
        for ref in references.GetAddedOrExplicitItems():
            asset_path = ref.assetPath
            if not asset_path or asset_path.startswith("@"):
                # Already-asset-path form or anonymous layer; skip.
                continue
            if "://" in asset_path or asset_path.startswith(("omniverse:", "/")):
                # Absolute URL / Omniverse Nucleus path → cannot resolve to a
                # file beside the packaged layer.
                return True
            resolved = (layer_dir / asset_path).resolve()
            if not resolved.exists():
                return True
        return False

    to_remove = [
        prim.GetPath()
        for prim in stage.Traverse()
        if _unresolvable_reference(prim)
    ]
    for path in to_remove:
        # RemovePrim works on the composed stage; for a prim that lives in the
        # root layer this deletes its spec so the reference stops dangling.
        stage.RemovePrim(path)

    if to_remove:
        stage.GetRootLayer().Save()
    return len(to_remove)
