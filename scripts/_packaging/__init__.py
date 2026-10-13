"""Shared packaging/gating library for the ``prepare_*_assets.py`` packagers.

Everything per-family lives in its packager script; everything the four
Genesis sanity gates shared — the near-verbatim helper copies, the catalog
writer, the asset-completeness check and the two ``sanity_check_*`` gates —
lives here once.  Importing the package pulls the submodules in::

    from _packaging.gate_common import ...      # helpers
    from _packaging.catalog import write_catalog
    from _packaging.usd_verify  import verify_object_assets
    from _packaging.sanity      import sanity_check_articulated, run_sanity_gate

(The leading underscore keeps ``import packaging`` — the installed PyPI
``packaging`` distribution that pytest/other libs rely on — untouched.)
"""

from __future__ import annotations

# genesis_compat.install() — centralised (architecture review §3.2): the
# packagers, the batch driver and the gate worker all import this package, so
# the shim is ambient for every packaging/gate path.  install() is idempotent
# (``genesis_compat._INSTALLED``); it degrades to a no-op without genesis.
try:
    from unienv_genesis_collection import genesis_compat

    genesis_compat.install()
except Exception:  # noqa: BLE001 — the shim degrades to a no-op without genesis
    pass

from . import _gate_batch, catalog, gate_common, sanity, usd_verify  # noqa: F401

__all__ = ["catalog", "gate_common", "sanity", "usd_verify", "_gate_batch"]
