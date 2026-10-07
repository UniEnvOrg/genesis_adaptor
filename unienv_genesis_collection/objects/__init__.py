"""Manipulatable objects for Genesis simulation scenes.

This package provides:

* :class:`ManipulatableObject` — abstract base for any object a robot can
  interact with.  Standalone descriptor (not a ``WorldNode``).
* Concrete primitive types: :class:`BoxObject`, :class:`SphereObject`,
  :class:`CylinderObject`.
* :class:`MeshObject` — mesh-file-backed object with trimesh-based geometry
  queries.
* :class:`ArticulatedObject` — multi-link furniture descriptor with an
  offline (FK-based) reference-configuration union bounding box, runtime
  joint/link name resolution (:func:`resolve_joint_dofs`,
  :func:`resolve_link_index`, raising :class:`ArticulationError`), an
  open-fraction helper (:meth:`ArticulatedObject.get_open_fraction`) and
  part-aware grasp candidates on named links.  Offline FK helpers:
  :func:`urdf_link_meshes_at_config`, :func:`mjcf_link_meshes_at_config`,
  :func:`usd_link_meshes_at_config`.
* :class:`YCBObject` — YCB dataset object (URDF with precomputed collision
  decomposition), plus :func:`get_ycb_catalog` and pool helpers
  :func:`list_ycb_objects` / :func:`make_ycb_pool`.
* :class:`GSOObject` — Google Scanned Objects entry (MJCF with V-HACD
  collision pieces), plus :func:`get_gso_catalog` and pool helpers
  :func:`list_gso_objects` / :func:`make_gso_pool`.
* :class:`ABOObject` — Amazon Berkeley Objects entry (CC BY 4.0; Z-up OBJ
  converted from the dataset's Y-up GLB, texture bound explicitly, collision
  from Genesis's own convex decomposition), plus :func:`get_abo_catalog` and
  pool helpers :func:`list_abo_objects` / :func:`make_abo_pool`.
* Hub-backed articulated furniture
  (:class:`~unienv_genesis_collection.objects.hub_objects.articulated.HubAssetArticulatedObject`
  base): :class:`RoboCasaFixture` (RoboCasa kitchen fixtures via
  ``gs.morphs.MJCF``), :class:`ReplicaCADFurniture` (ReplicaCAD furniture
  via ``gs.morphs.URDF``), :class:`ArtVIPFurniture` (ArtVIP furniture via
  ``gs.morphs.USD`` + ``scene.add_stage``), :class:`MolmoSpacesFurniture`
  (MolmoSpaces Ai2-THOR subset, USD), :class:`LightwheelFurniture`
  (Lightwheel SimReady bundle, USD) and :class:`SimReadyWarehouseProp`
  (NVIDIA SimReady Warehouse **rigid props**, USD), each with catalog
  helpers (``get_robocasa_catalog`` / ``list_robocasa_objects`` /
  ``make_robocasa_pool`` and the ``replicacad`` / ``artvip`` /
  ``molmospaces`` / ``lightwheel`` / ``simready_warehouse`` equivalents).
* :class:`LightwheelEnv` — the Lightwheel bundle's **static** ``Locomotion/``
  environments (terrains/rooms, USD via ``gs.morphs.USD`` +
  ``scene.add_stage``), the non-articulated sibling of
  :class:`LightwheelFurniture`, with the ``lightwheel_env`` catalog helpers
  (CC BY-NC 4.0 — see ``docs/articulated_rooms.md``).
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
from unienv_genesis_collection.objects.articulated_object import (
    ArticulatedObject,
    ArticulationError,
    resolve_joint_dofs,
    resolve_link_index,
    urdf_link_meshes_at_config,
    mjcf_link_meshes_at_config,
    usd_link_meshes_at_config,
)
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
    ABOObject,
    get_abo_catalog,
    list_abo_objects,
    make_abo_pool,
    HubAssetArticulatedObject,
    RoboCasaFixture,
    get_robocasa_catalog,
    list_robocasa_objects,
    make_robocasa_pool,
    ReplicaCADFurniture,
    get_replicacad_catalog,
    list_replicacad_objects,
    make_replicacad_pool,
    ArtVIPFurniture,
    get_artvip_catalog,
    list_artvip_objects,
    make_artvip_pool,
    MolmoSpacesFurniture,
    get_molmospaces_catalog,
    list_molmospaces_objects,
    make_molmospaces_pool,
    LightwheelFurniture,
    get_lightwheel_catalog,
    list_lightwheel_objects,
    make_lightwheel_pool,
    SimReadyWarehouseProp,
    get_simready_warehouse_catalog,
    list_simready_warehouse_objects,
    make_simready_warehouse_pool,
    LightwheelEnv,
    get_lightwheel_env_catalog,
    list_lightwheel_env_objects,
    make_lightwheel_env_pool,
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
    # Articulated furniture
    "ArticulatedObject",
    "ArticulationError",
    "resolve_joint_dofs",
    "resolve_link_index",
    "urdf_link_meshes_at_config",
    "mjcf_link_meshes_at_config",
    "usd_link_meshes_at_config",
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
    # Amazon Berkeley Objects
    "ABOObject",
    "get_abo_catalog",
    "list_abo_objects",
    "make_abo_pool",
    # Articulated furniture (Hub-backed)
    "HubAssetArticulatedObject",
    # RoboCasa kitchen fixtures (MJCF)
    "RoboCasaFixture",
    "get_robocasa_catalog",
    "list_robocasa_objects",
    "make_robocasa_pool",
    # ReplicaCAD furniture (URDF)
    "ReplicaCADFurniture",
    "get_replicacad_catalog",
    "list_replicacad_objects",
    "make_replicacad_pool",
    # ArtVIP furniture (USD)
    "ArtVIPFurniture",
    "get_artvip_catalog",
    "list_artvip_objects",
    "make_artvip_pool",
    # MolmoSpaces furniture (USD)
    "MolmoSpacesFurniture",
    "get_molmospaces_catalog",
    "list_molmospaces_objects",
    "make_molmospaces_pool",
    # Lightwheel furniture (USD)
    "LightwheelFurniture",
    "get_lightwheel_catalog",
    "list_lightwheel_objects",
    "make_lightwheel_pool",
    # SimReady Warehouse rigid props (USD)
    "SimReadyWarehouseProp",
    "get_simready_warehouse_catalog",
    "list_simready_warehouse_objects",
    "make_simready_warehouse_pool",
    # Lightwheel static environments (USD, Locomotion subset)
    "LightwheelEnv",
    "get_lightwheel_env_catalog",
    "list_lightwheel_env_objects",
    "make_lightwheel_env_pool",
    # Tool
    "ManipulatableTool",
    "FunctionalPoint",
    # Container
    "ManipulatableContainer",
]
