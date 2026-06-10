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
]
