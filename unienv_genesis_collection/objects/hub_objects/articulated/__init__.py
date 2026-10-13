"""Hub-backed articulated furniture descriptors (plan §1.4).

This subpackage extends the Hub-dataset object pattern (``YCBObject`` /
``GSOObject``) to *articulated* furniture.  The base class reimplements the
``HubAssetMeshObject`` machinery (asset directory resolution, required-file
enumeration, lazy verified downloads) without inheriting ``MeshObject``'s
single-mesh assumptions, and the three concrete datasets provide the
format-specific offline FK and scene loading:

* :class:`HubAssetArticulatedObject` — shared base: offline-safe
  construction, ``_required_files`` (MJCF ``file=`` + recursive
  ``<include>`` chain, URDF ``filename=``, USD pxr layer stack +
  ``resource/`` textures), ``_ensure_assets`` / ``ensure_assets``.
* :class:`RoboCasaFixture` — RoboCasa kitchen fixtures (MJCF,
  ``gs.morphs.MJCF``; fixed = jointless root), with
  :func:`get_robocasa_catalog` / :func:`list_robocasa_objects` /
  :func:`make_robocasa_pool`.
* :class:`ReplicaCADFurniture` — ReplicaCAD furniture (URDF,
  ``gs.morphs.URDF``, ``merge_fixed_links=False``), with
  :func:`get_replicacad_catalog` / :func:`list_replicacad_objects` /
  :func:`make_replicacad_pool`.
* :class:`ArtVIPFurniture` — ArtVIP furniture (USD, ``gs.morphs.USD`` +
  ``scene.add_stage``, single-entity assert, clean-name translation of
  Genesis's full-path USD joint/link names), with
  :func:`get_artvip_catalog` / :func:`list_artvip_objects` /
  :func:`make_artvip_pool`.
* :class:`MolmoSpacesFurniture` — MolmoSpaces (Ai2-THOR) furniture (USD,
  the same ``add_stage`` contract; the packaged stages carry the mandatory
  joint-body0/body1 swap), with :func:`get_molmospaces_catalog` /
  :func:`list_molmospaces_objects` / :func:`make_molmospaces_pool`.
* :class:`LightwheelFurniture` — Lightwheel SimReady furniture (USD, the
  same ``add_stage`` contract; packaged stages carry the MDL →
  UsdPreviewSurface material rewrite), with :func:`get_lightwheel_catalog` /
  :func:`list_lightwheel_objects` / :func:`make_lightwheel_pool`.
* :class:`SimReadyWarehouseProp` — NVIDIA SimReady Warehouse **rigid
  props** (USD via ``scene.add_stage``; ``RigidBodyAPI`` authored at
  packaging, zero joints — an empty-joint member of the family), with
  :func:`get_simready_warehouse_catalog` /
  :func:`list_simready_warehouse_objects` /
  :func:`make_simready_warehouse_pool`.
"""

from unienv_genesis_collection.objects.hub_objects.articulated.hub_articulated_object import (
    HubAssetArticulatedObject,
)
from unienv_genesis_collection.objects.hub_objects.articulated.robocasa_fixture import (
    ROBOCASA_ASSETS_ROOT,
    ROBOCASA_CATALOG_PATH,
    RoboCasaFixture,
    get_robocasa_catalog,
    list_robocasa_objects,
    make_robocasa_pool,
)
from unienv_genesis_collection.objects.hub_objects.articulated.replicacad_furniture import (
    REPLICACAD_ASSETS_ROOT,
    REPLICACAD_CATALOG_PATH,
    ReplicaCADFurniture,
    get_replicacad_catalog,
    list_replicacad_objects,
    make_replicacad_pool,
)
from unienv_genesis_collection.objects.hub_objects.articulated.artvip_furniture import (
    ARTVIP_ASSETS_ROOT,
    ARTVIP_CATALOG_PATH,
    ArtVIPFurniture,
    get_artvip_catalog,
    list_artvip_objects,
    make_artvip_pool,
)
from unienv_genesis_collection.objects.hub_objects.articulated.molmospaces_furniture import (
    MOLMOSPACES_ASSETS_ROOT,
    MOLMOSPACES_CATALOG_PATH,
    MolmoSpacesFurniture,
    get_molmospaces_catalog,
    list_molmospaces_objects,
    make_molmospaces_pool,
)
from unienv_genesis_collection.objects.hub_objects.articulated.lightwheel_furniture import (
    LIGHTWHEEL_ASSETS_ROOT,
    LIGHTWHEEL_CATALOG_PATH,
    LightwheelFurniture,
    get_lightwheel_catalog,
    list_lightwheel_objects,
    make_lightwheel_pool,
)
from unienv_genesis_collection.objects.hub_objects.articulated.simready_warehouse import (
    SIMREADY_WAREHOUSE_ASSETS_ROOT,
    SIMREADY_WAREHOUSE_CATALOG_PATH,
    SimReadyWarehouseProp,
    get_simready_warehouse_catalog,
    list_simready_warehouse_objects,
    make_simready_warehouse_pool,
)

__all__ = [
    # Base
    "HubAssetArticulatedObject",
    # RoboCasa fixtures (MJCF)
    "RoboCasaFixture",
    "ROBOCASA_ASSETS_ROOT",
    "ROBOCASA_CATALOG_PATH",
    "get_robocasa_catalog",
    "list_robocasa_objects",
    "make_robocasa_pool",
    # ReplicaCAD furniture (URDF)
    "ReplicaCADFurniture",
    "REPLICACAD_ASSETS_ROOT",
    "REPLICACAD_CATALOG_PATH",
    "get_replicacad_catalog",
    "list_replicacad_objects",
    "make_replicacad_pool",
    # ArtVIP furniture (USD)
    "ArtVIPFurniture",
    "ARTVIP_ASSETS_ROOT",
    "ARTVIP_CATALOG_PATH",
    "get_artvip_catalog",
    "list_artvip_objects",
    "make_artvip_pool",
    # MolmoSpaces furniture (USD, joint-body swap applied at packaging)
    "MolmoSpacesFurniture",
    "MOLMOSPACES_ASSETS_ROOT",
    "MOLMOSPACES_CATALOG_PATH",
    "get_molmospaces_catalog",
    "list_molmospaces_objects",
    "make_molmospaces_pool",
    # Lightwheel furniture (USD, MDL → UsdPreviewSurface at packaging)
    "LightwheelFurniture",
    "LIGHTWHEEL_ASSETS_ROOT",
    "LIGHTWHEEL_CATALOG_PATH",
    "get_lightwheel_catalog",
    "list_lightwheel_objects",
    "make_lightwheel_pool",
    # SimReady Warehouse rigid props (USD, RigidBodyAPI authored at packaging)
    "SimReadyWarehouseProp",
    "SIMREADY_WAREHOUSE_ASSETS_ROOT",
    "SIMREADY_WAREHOUSE_CATALOG_PATH",
    "get_simready_warehouse_catalog",
    "list_simready_warehouse_objects",
    "make_simready_warehouse_pool",
]
