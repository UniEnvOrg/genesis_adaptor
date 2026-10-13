"""Collection utility probes (pxr converters for asset packaging).

The :mod:`unienv_genesis_collection.utils.usd_conversions` module holds the
offline OpenUSD converter recipes shared by the asset-prep scripts; this
package shim keeps them out of the runtime import path (every ``pxr`` import
inside stays lazy, so importing :mod:`unienv_genesis_collection` is cheap for
non-USD users).
"""
