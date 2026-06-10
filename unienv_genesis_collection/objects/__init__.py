"""Manipulatable objects for Genesis simulation scenes.

This package provides:

* :class:`ManipulatableObject` — abstract base for any object a robot can
  interact with.  Standalone descriptor (not a ``WorldNode``).
* Concrete primitive types: :class:`BoxObject`, :class:`SphereObject`,
  :class:`CylinderObject`.
* :class:`MeshObject` — mesh-file-backed object with trimesh-based geometry
  queries.
* :class:`YCBObject` — YCB dataset object (URDF with precomputed collision
  decomposition), plus :func:`get_ycb_catalog` and pool helpers
  :func:`list_ycb_objects` / :func:`make_ycb_pool`.
* :class:`GSOObject` — Google Scanned Objects entry (MJCF with V-HACD
  collision pieces), plus :func:`get_gso_catalog` and pool helpers
  :func:`list_gso_objects` / :func:`make_gso_pool`.
* :class:`ObjectInfo` — per-object catalog metadata (language annotations
  ordered broad → detailed, and category tags in a separate field); dataset
  catalogs ship with the assets as ``catalog.json``.
* :class:`ManipulatableTool` — mixin ABC for objects with tool functionality
  (sweeping, scooping, cutting, etc.).
* :class:`ManipulatableContainer` — abstract base for hollow objects (bowls,
  cups, trays).
* Helper functions for working with bounding boxes represented as
  ``BoxSpace``: :func:`make_bbox`, :func:`bbox_size`, :func:`bbox_center`,
  :func:`bbox_half_extents`.
"""

from unienv_genesis_collection.objects.base_object import (
    ManipulatableObject,
    EntityT,
    GraspCandidates,
    make_bbox,
    bbox_size,
    bbox_center,
    bbox_half_extents,
)
from unienv_genesis_collection.objects.primitive_objects import (
    BoxObject,
    SphereObject,
    CylinderObject,
)
from unienv_genesis_collection.objects.mesh_object import MeshObject
from unienv_genesis_collection.objects.hub_objects import (
    ObjectInfo,
    YCBObject,
    get_ycb_catalog,
    list_ycb_objects,
    make_ycb_pool,
    GSOObject,
    get_gso_catalog,
    list_gso_objects,
    make_gso_pool,
)
from unienv_genesis_collection.objects.base_tool import (
    ManipulatableTool,
    FunctionalPoint,
)
from unienv_genesis_collection.objects.base_container import (
    ManipulatableContainer,
)

__all__ = [
    # Base + helpers
    "ManipulatableObject",
    "EntityT",
    "GraspCandidates",
    "make_bbox",
    "bbox_size",
    "bbox_center",
    "bbox_half_extents",
    # Primitives
    "BoxObject",
    "SphereObject",
    "CylinderObject",
    # Mesh
    "MeshObject",
    # Dataset catalogs
    "ObjectInfo",
    # YCB
    "YCBObject",
    "get_ycb_catalog",
    "list_ycb_objects",
    "make_ycb_pool",
    # Google Scanned Objects
    "GSOObject",
    "get_gso_catalog",
    "list_gso_objects",
    "make_gso_pool",
    # Tool
    "ManipulatableTool",
    "FunctionalPoint",
    # Container
    "ManipulatableContainer",
]
