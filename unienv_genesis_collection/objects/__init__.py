"""Manipulatable objects for Genesis simulation scenes.

This package provides:

* :class:`ManipulatableObject` — abstract base for any object a robot can
  interact with.  Standalone descriptor (not a ``WorldNode``).
* Concrete primitive types: :class:`BoxObject`, :class:`SphereObject`,
  :class:`CylinderObject`.
* :class:`MeshObject` — mesh-file-backed object with trimesh-based geometry
  queries.
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
from unienv_genesis_collection.objects.pick_place_objs import YCBManipulatableObject
from unienv_genesis_collection.objects.ycb_parser import YCBParser
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
    "YCBManipulatableObject",
    "YCBParser",
    # Tool
    "ManipulatableTool",
    "FunctionalPoint",
    # Container
    "ManipulatableContainer",
]
