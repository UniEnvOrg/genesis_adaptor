"""YCB dataset objects as :class:`ManipulatableObject` descriptors.

The assets are a repackaged copy of `elpis-lab/YCB_Dataset
<https://github.com/elpis-lab/YCB_Dataset>`_ (MIT licensed; Google-16K scans
from the `YCB benchmark <https://www.ycbbenchmarks.com/>`_), hosted on the
collection's Hugging Face assets repo under ``objects/ycb/<object_id>/``.
Each object directory is self-contained::

    objects/ycb/banana/banana.urdf       # visual + precomputed CoACD collision
    objects/ycb/banana/banana.xml        # MJCF equivalent (not used here)
    objects/ycb/banana/textured.obj      # + textured.mtl, texture_map.png
    objects/ycb/banana/textured_coacd_*.stl

Objects are loaded into Genesis via ``gs.morphs.URDF``, which uses the
precomputed convex decomposition and the real object mass baked into the
URDF.  Geometry queries (bounding box, grasp sampling) reuse the
trimesh-based machinery of :class:`MeshObject` on ``textured.obj`` — valid
because the URDF places the visual mesh at the link origin with no offset.

Assets are fetched lazily: the object directory is downloaded (and verified
for completeness) on first scene load or geometry query.  Per-object
metadata — language annotations (broad → detailed) and tags, kept in
separate fields — ships with the assets as ``objects/ycb/catalog.json``
(a single small file, fetched on first catalog access).

Typical usage::

    from unienv_genesis_collection.objects import YCBObject, make_ycb_pool

    banana = YCBObject("banana")
    pool = make_ycb_pool(tags="food", n=5,
                         rng=torch.Generator().manual_seed(0))
"""

from __future__ import annotations

from pathlib import Path
from typing import (
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    TYPE_CHECKING,
)

import torch

from unienv_genesis_collection.objects.hub_objects.catalog import (
    ObjectInfo,
    load_catalog,
)
from unienv_genesis_collection.objects.hub_objects.hub_asset_object import (
    HubAssetMeshObject,
    xml_file_references,
)

if TYPE_CHECKING:
    from unienv_genesis.world.genesis_world import GenesisWorld

__all__ = [
    "YCB_ASSETS_ROOT",
    "YCB_CATALOG_PATH",
    "get_ycb_catalog",
    "YCBObject",
    "list_ycb_objects",
    "make_ycb_pool",
]

# Path of the YCB collection relative to the assets root (POSIX separators).
YCB_ASSETS_ROOT = "objects/ycb"


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

# The catalog (language annotations + tags per object) ships with the assets
# as ``objects/ycb/catalog.json``; its curated source lives in
# ``scripts/data/ycb_catalog.json``.
YCB_CATALOG_PATH = f"{YCB_ASSETS_ROOT}/catalog.json"


def get_ycb_catalog() -> Dict[str, ObjectInfo]:
    """Return the YCB catalog (object id → :class:`ObjectInfo`).

    Loaded from ``objects/ycb/catalog.json`` in the assets folder
    (downloaded on first use) and cached.  Annotations are ordered from
    broad to detailed; tags include categories (``food``, ``tool``, ...)
    plus ``box_gripper_graspable`` when the object's smallest mesh
    bounding-box dimension fits a ~8.5 cm parallel-jaw gripper (computed by
    ``scripts/compute_graspable.py``; for a per-pose mesh check at runtime
    use :meth:`ManipulatableObject.is_box_gripper_graspable`).
    """
    return load_catalog(YCB_CATALOG_PATH)


def list_ycb_objects(*, tags: Optional[Union[str, Iterable[str]]] = None) -> List[str]:
    """Return YCB object ids, optionally filtered by tags.

    Consults only the catalog (fetched once, then cached) — no object
    assets are downloaded.

    Args:
        tags: A tag or iterable of tags.  Only objects carrying **all** of
            them are returned.  ``None`` returns every object.

    Returns:
        Sorted list of object ids.
    """
    if tags is None:
        wanted: FrozenSet[str] = frozenset()
    elif isinstance(tags, str):
        wanted = frozenset((tags,))
    else:
        wanted = frozenset(tags)
    return sorted(
        object_id
        for object_id, info in get_ycb_catalog().items()
        if wanted <= info.tags
    )


# ---------------------------------------------------------------------------
# YCBObject
# ---------------------------------------------------------------------------

class YCBObject(HubAssetMeshObject):
    """A YCB object loaded into Genesis via its prepared URDF.

    Inherits the trimesh-based bounding-box and grasp-candidate machinery
    from :class:`MeshObject` (pointed at the object's ``textured.obj``),
    but overrides :meth:`load_into_scene` to load the URDF so Genesis uses
    the precomputed CoACD collision hulls and the real object mass.

    Construction is offline-safe; assets are downloaded lazily on first
    scene load or geometry query and verified for completeness (the URDF
    references several files which must all be present locally).

    Args:
        object_id: Catalog id, e.g. ``"banana"``.  See
            :func:`get_ycb_catalog` or :func:`list_ycb_objects`.
        scale: Uniform scalar or per-axis ``(sx, sy, sz)`` scale factor.
        name: Technical identifier; defaults to *object_id*.
        fixed: Whether the object is immovable.
        language_annotations: Override the catalog's default annotations.
        standing_quaternion: Optional ``(w, x, y, z)`` orientation used by
            :meth:`set_pos_3d` instead of the identity when the object is
            placed without randomisation (e.g. to stand a drill on its
            battery pack).  Randomised placement applies a random yaw on
            top of it.  Defaults to the catalog's ``standing_quaternion``
            metadata when present; pass ``(1, 0, 0, 0)`` to force the
            identity orientation for an object whose catalog entry defines
            one.
    """

    def __init__(
        self,
        object_id: str,
        scale: Union[float, Tuple[float, float, float]] = 1.0,
        name: Optional[str] = None,
        fixed: bool = False,
        language_annotations: Optional[Sequence[str]] = None,
        standing_quaternion: Optional[Sequence[float]] = None,
    ):
        catalog = get_ycb_catalog()
        info = catalog.get(object_id)
        if info is None:
            raise KeyError(
                f"Unknown YCB object id: {object_id!r}. "
                f"See list_ycb_objects() for the {len(catalog)} available ids."
            )
        super().__init__(
            asset_relative_dir=f"{YCB_ASSETS_ROOT}/{object_id}",
            mesh_file_name="textured.obj",
            scale=scale,
            name=name or object_id,
            fixed=fixed,
            language_annotations=(
                language_annotations
                if language_annotations is not None
                else info.language_annotations
            ),
            visual_texture_file_name="texture_map.png",
            standing_quaternion=(
                standing_quaternion
                if standing_quaternion is not None
                else info.standing_quaternion
            ),
        )
        self.object_id = object_id
        self.info = info
        self.urdf_file = self.asset_dir / f"{object_id}.urdf"
        # Scalar scales are passed to the morph as float, per-axis as tuple.
        self._morph_scale: Union[float, Tuple[float, float, float]] = (
            scale if isinstance(scale, tuple) else float(scale)
        )

    def _required_files(self) -> List[Path]:
        """The URDF, visual mesh + texture, and every URDF-referenced mesh
        (the number of CoACD collision pieces varies per object)."""
        required = [
            self.urdf_file,
            self.asset_dir / "textured.obj",
            self.asset_dir / "textured.mtl",
            self.asset_dir / "texture_map.png",
        ]
        if self.urdf_file.is_file():
            required += [
                self.asset_dir / ref
                for ref in xml_file_references(self.urdf_file, attribute="filename")
            ]
        return required

    def load_into_scene(
        self,
        world: "GenesisWorld",
        pos=(0.0, 0.0, 0.0),
        quat=(1.0, 0.0, 0.0, 0.0),
        name_override=None,
        batch_fixed_verts=False,
    ):
        import genesis as gs

        self._ensure_assets()
        morph = gs.morphs.URDF(
            file=str(self.urdf_file),
            scale=self._morph_scale,
            pos=pos,
            quat=quat,
            align=False,
            fixed=self.fixed,
            batch_fixed_verts=batch_fixed_verts,
            # Nyx currently cannot export URDF entities after Genesis merges
            # fixed links, so keep the original link structure for YCB assets.
            merge_fixed_links=False,
        )
        # No surface override: Default() preserves the texture from the asset.
        return world.scene.add_entity(
            morph,
            name=name_override or self.name,
        )


# ---------------------------------------------------------------------------
# Pool building
# ---------------------------------------------------------------------------

def make_ycb_pool(
    object_ids: Optional[Sequence[str]] = None,
    *,
    tags: Optional[Union[str, Iterable[str]]] = None,
    n: Optional[int] = None,
    rng: Optional[torch.Generator] = None,
    **object_kwargs,
) -> List[YCBObject]:
    """Build a list of :class:`YCBObject` for use as an object pool.

    The result plugs directly into consumers such as
    ``ClutteredTableManager(object_pool=...)``.  Construction does not
    download anything; assets are fetched when the objects are first
    loaded into a scene.

    Args:
        object_ids: Explicit ids to use (order preserved).  Mutually
            exclusive with *tags*.
        tags: Filter the catalog by tags (see :func:`list_ycb_objects`).
        n: If given, sample *n* distinct objects from the selection.
        rng: Optional :class:`torch.Generator` for reproducible sampling.
            When omitted (``None``), the global Torch RNG is used.
        **object_kwargs: Forwarded to every :class:`YCBObject` (e.g.
            ``scale=0.8``).

    Returns:
        List of :class:`YCBObject` descriptors with unique names.

    Raises:
        ValueError: If both *object_ids* and *tags* are given, or *n*
            exceeds the number of selected objects.
        KeyError: If an explicit id is not in the catalog.
    """
    if object_ids is not None and tags is not None:
        raise ValueError("Pass either object_ids or tags, not both.")

    ids = list(object_ids) if object_ids is not None else list_ycb_objects(tags=tags)

    if n is not None:
        if n > len(ids):
            raise ValueError(
                f"Requested {n} objects but only {len(ids)} match the selection."
            )
        # Pin the output device to the generator's device: with a Genesis
        # CUDA default device, an unpinned randperm would create a CUDA tensor
        # from a CPU generator and raise a device-mismatch error.
        perm = torch.randperm(
            len(ids), generator=rng, device=rng.device if rng is not None else None
        )[:n].tolist()
        ids = [ids[i] for i in perm]

    return [YCBObject(object_id, **object_kwargs) for object_id in ids]
