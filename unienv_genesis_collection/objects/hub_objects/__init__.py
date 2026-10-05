"""Manipulatable objects backed by datasets hosted on the Hugging Face Hub.

These objects are multi-file assets (description file + meshes + textures)
that live on the collection's HF assets repo and are downloaded lazily into
the local assets folder.  This subpackage provides:

* :class:`HubAssetMeshObject` — base class with the shared lazy, verified
  asset-fetch machinery.
* :class:`ObjectInfo` / :func:`load_catalog` — per-object metadata
  (language annotations ordered broad → detailed, and category tags) loaded
  from a dataset's ``catalog.json`` shipped alongside its assets.
* :class:`YCBObject` — YCB dataset object (URDF + precomputed CoACD
  collision), with :func:`get_ycb_catalog`, :func:`list_ycb_objects`,
  :func:`make_ycb_pool`.
* :class:`GSOObject` — Google Scanned Objects entry (MJCF + V-HACD
  collision), with :func:`get_gso_catalog`, :func:`list_gso_objects`,
  :func:`make_gso_pool`.
* The :mod:`~unienv_genesis_collection.objects.hub_objects.articulated`
  subpackage — Hub-backed *articulated* furniture:
  :class:`~...articulated.HubAssetArticulatedObject` base plus
  :class:`~...articulated.RoboCasaFixture` (MJCF),
  :class:`~...articulated.ReplicaCADFurniture` (URDF),
  :class:`~...articulated.ArtVIPFurniture` (USD),
  :class:`~...articulated.MolmoSpacesFurniture` (USD, Ai2-THOR subset),
  :class:`~...articulated.LightwheelFurniture` (SimReady bundle) and
  :class:`~...articulated.SimReadyWarehouseProp` (rigid props) with their
  catalog / list / pool helpers.
* :class:`LightwheelEnv` — the Lightwheel SimReady bundle's **static**
  ``Locomotion/`` environments (terrains, rooms, a staircase; loaded with
  ``gs.morphs.USD`` + ``scene.add_stage``, no articulation), with
  :func:`get_lightwheel_env_catalog`, :func:`list_lightwheel_env_objects`,
  :func:`make_lightwheel_env_pool`.
"""

from unienv_genesis_collection.objects.hub_objects.catalog import (
    ObjectInfo,
    load_catalog,
)
from unienv_genesis_collection.objects.hub_objects.hub_asset_object import (
    HubAssetMeshObject,
    xml_file_references,
)
from unienv_genesis_collection.objects.hub_objects.ycb_object import (
    YCBObject,
    YCB_ASSETS_ROOT,
    YCB_CATALOG_PATH,
    get_ycb_catalog,
    list_ycb_objects,
    make_ycb_pool,
)
from unienv_genesis_collection.objects.hub_objects.gso_object import (
    GSOObject,
    GSO_ASSETS_ROOT,
    GSO_CATALOG_PATH,
    get_gso_catalog,
    list_gso_objects,
    make_gso_pool,
)
from unienv_genesis_collection.objects.hub_objects.articulated import (
    HubAssetArticulatedObject,
    ROBOCASA_ASSETS_ROOT,
    ROBOCASA_CATALOG_PATH,
    RoboCasaFixture,
    get_robocasa_catalog,
    list_robocasa_objects,
    make_robocasa_pool,
    REPLICACAD_ASSETS_ROOT,
    REPLICACAD_CATALOG_PATH,
    ReplicaCADFurniture,
    get_replicacad_catalog,
    list_replicacad_objects,
    make_replicacad_pool,
    ARTVIP_ASSETS_ROOT,
    ARTVIP_CATALOG_PATH,
    ArtVIPFurniture,
    get_artvip_catalog,
    list_artvip_objects,
    make_artvip_pool,
    MOLMOSPACES_ASSETS_ROOT,
    MOLMOSPACES_CATALOG_PATH,
    MolmoSpacesFurniture,
    get_molmospaces_catalog,
    list_molmospaces_objects,
    make_molmospaces_pool,
    LIGHTWHEEL_ASSETS_ROOT,
    LIGHTWHEEL_CATALOG_PATH,
    LightwheelFurniture,
    get_lightwheel_catalog,
    list_lightwheel_objects,
    make_lightwheel_pool,
    SIMREADY_WAREHOUSE_ASSETS_ROOT,
    SIMREADY_WAREHOUSE_CATALOG_PATH,
    SimReadyWarehouseProp,
    get_simready_warehouse_catalog,
    list_simready_warehouse_objects,
    make_simready_warehouse_pool,
)
from unienv_genesis_collection.objects.hub_objects.lightwheel_env import (
    LIGHTWHEEL_ENV_ASSETS_ROOT,
    LIGHTWHEEL_ENV_CATALOG_PATH,
    LightwheelEnv,
    get_lightwheel_env_catalog,
    list_lightwheel_env_objects,
    make_lightwheel_env_pool,
)

__all__ = [
    # Base + catalog
    "HubAssetMeshObject",
    "xml_file_references",
    "ObjectInfo",
    "load_catalog",
    # YCB
    "YCBObject",
    "YCB_ASSETS_ROOT",
    "YCB_CATALOG_PATH",
    "get_ycb_catalog",
    "list_ycb_objects",
    "make_ycb_pool",
    # Google Scanned Objects
    "GSOObject",
    "GSO_ASSETS_ROOT",
    "GSO_CATALOG_PATH",
    "get_gso_catalog",
    "list_gso_objects",
    "make_gso_pool",
    # Articulated furniture (RoboCasa MJCF / ReplicaCAD URDF)
    "HubAssetArticulatedObject",
    "RoboCasaFixture",
    "ROBOCASA_ASSETS_ROOT",
    "ROBOCASA_CATALOG_PATH",
    "get_robocasa_catalog",
    "list_robocasa_objects",
    "make_robocasa_pool",
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
    # MolmoSpaces furniture (USD)
    "MolmoSpacesFurniture",
    "MOLMOSPACES_ASSETS_ROOT",
    "MOLMOSPACES_CATALOG_PATH",
    "get_molmospaces_catalog",
    "list_molmospaces_objects",
    "make_molmospaces_pool",
    # Lightwheel furniture (USD)
    "LightwheelFurniture",
    "LIGHTWHEEL_ASSETS_ROOT",
    "LIGHTWHEEL_CATALOG_PATH",
    "get_lightwheel_catalog",
    "list_lightwheel_objects",
    "make_lightwheel_pool",
    # SimReady Warehouse rigid props (USD)
    "SimReadyWarehouseProp",
    "SIMREADY_WAREHOUSE_ASSETS_ROOT",
    "SIMREADY_WAREHOUSE_CATALOG_PATH",
    "get_simready_warehouse_catalog",
    "list_simready_warehouse_objects",
    "make_simready_warehouse_pool",
    # Lightwheel static environments (USD, Locomotion subset)
    "LightwheelEnv",
    "LIGHTWHEEL_ENV_ASSETS_ROOT",
    "LIGHTWHEEL_ENV_CATALOG_PATH",
    "get_lightwheel_env_catalog",
    "list_lightwheel_env_objects",
    "make_lightwheel_env_pool",
]
