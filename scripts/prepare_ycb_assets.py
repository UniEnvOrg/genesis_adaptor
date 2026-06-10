#!/usr/bin/env python3
"""Prepare YCB assets for the collection's Hugging Face assets repo.

Clones (or reuses) `elpis-lab/YCB_Dataset
<https://github.com/elpis-lab/YCB_Dataset>`_ and reorganizes it into one
self-contained directory per object, the layout expected by
``unienv_genesis_collection.objects.hub_objects.ycb_object``::

    <output>/objects/ycb/LICENSE          # upstream MIT license, verbatim
    <output>/objects/ycb/README.md        # provenance + attribution
    <output>/objects/ycb/catalog.json     # curated annotations + tags
    <output>/objects/ycb/banana/banana.urdf
    <output>/objects/ycb/banana/banana.xml
    <output>/objects/ycb/banana/textured.obj  (+ .mtl, texture_map.png, coacd STLs)

Upstream URDF/MJCF files reference meshes as ``banana/textured.obj``
(relative to the repo's ``ycb/`` root); moving them inside each object
directory requires rewriting those references to be directory-local.

Usage::

    python scripts/prepare_ycb_assets.py --output-dir /tmp/ycb_assets
    # or with an existing clone:
    python scripts/prepare_ycb_assets.py --source ~/YCB_Dataset --output-dir /tmp/ycb_assets

Then upload to the Hub (needs write access to the org)::

    hf upload UniEnvOrg/unienv_genesis_collection_assets \
        /tmp/ycb_assets/objects/ycb objects/ycb --repo-type dataset
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

UPSTREAM_REPO_URL = "https://github.com/elpis-lab/YCB_Dataset.git"

# Curated per-object metadata (annotations + tags), shipped with the assets
# as ``objects/ycb/catalog.json``.
_CATALOG_SOURCE = Path(__file__).parent / "data" / "ycb_catalog.json"

PROVENANCE_README = """\
# YCB objects

Repackaged from [elpis-lab/YCB_Dataset](https://github.com/elpis-lab/YCB_Dataset)
(MIT licensed, see `LICENSE`): Google-16K scans from the
[YCB Object and Model Set](https://www.ycbbenchmarks.com/), OBB-aligned with
origin at the mass center, with CoACD convex decompositions and real object
masses baked into the URDFs.

Reorganized to one self-contained directory per object (URDF/MJCF mesh paths
rewritten to be directory-local) by
`genesis_adaptor/scripts/prepare_ycb_assets.py`.

When using these models, please cite the YCB benchmark:

> B. Calli, A. Singh, A. Walsman, S. Srinivasa, P. Abbeel, A. M. Dollar,
> "The YCB Object and Model Set: Towards Common Benchmarks for Manipulation
> Research", ICAR 2015.
"""


def rewrite_mesh_paths(text: str, object_id: str) -> str:
    """Rewrite ``filename="<id>/..."`` / ``file="<id>/..."`` references to be
    relative to the object's own directory."""
    return re.sub(
        rf'((?:filename|file)=")({re.escape(object_id)})/',
        r"\1",
        text,
    )


def _clone_upstream(dest: Path) -> Path:
    print(f"Cloning {UPSTREAM_REPO_URL} into {dest} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", UPSTREAM_REPO_URL, str(dest)],
        check=True,
    )
    return dest


def _verify_object_dir(obj_out_dir: Path, object_id: str) -> None:
    """Check that every mesh referenced by the rewritten URDF exists."""
    urdf = obj_out_dir / f"{object_id}.urdf"
    root = ET.parse(urdf).getroot()
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if filename and not (obj_out_dir / filename).is_file():
            raise FileNotFoundError(
                f"{urdf.name} references missing mesh: {filename}"
            )


def prepare(source: Path, output_dir: Path) -> Path:
    ycb_src = source / "ycb"
    if not ycb_src.is_dir():
        raise FileNotFoundError(f"No 'ycb' directory in source: {source}")

    out_root = output_dir / "objects" / "ycb"
    out_root.mkdir(parents=True, exist_ok=True)

    object_ids = sorted(p.name for p in ycb_src.iterdir() if p.is_dir())
    if not object_ids:
        raise FileNotFoundError(f"No object directories found in {ycb_src}")

    for object_id in object_ids:
        obj_out_dir = out_root / object_id
        obj_out_dir.mkdir(exist_ok=True)

        # Mesh / texture files, copied as-is.
        for f in sorted((ycb_src / object_id).iterdir()):
            if f.is_file():
                shutil.copy2(f, obj_out_dir / f.name)

        # URDF / MJCF with directory-local mesh paths.
        for suffix in (".urdf", ".xml"):
            src_file = ycb_src / f"{object_id}{suffix}"
            if not src_file.is_file():
                raise FileNotFoundError(f"Missing {src_file}")
            text = rewrite_mesh_paths(src_file.read_text(), object_id)
            (obj_out_dir / src_file.name).write_text(text)

        _verify_object_dir(obj_out_dir, object_id)

    shutil.copy2(source / "LICENSE", out_root / "LICENSE")
    (out_root / "README.md").write_text(PROVENANCE_README)
    _write_catalog(out_root, object_ids)

    print(f"Prepared {len(object_ids)} objects in {out_root}")
    return out_root


def _write_catalog(out_root: Path, object_ids: list[str]) -> None:
    """Copy the curated catalog into the prepared tree as ``catalog.json``.

    The curated source (language annotations, broad → detailed, plus
    category tags) lives in ``scripts/data/ycb_catalog.json``; entries are
    cross-checked against the prepared object directories.
    """
    catalog = json.loads(_CATALOG_SOURCE.read_text())
    for entry in catalog.values():
        assert entry["language_annotations"] and entry["tags"] is not None
    prepared = set(object_ids)
    for missing in sorted(set(catalog) - prepared):
        print(f"WARNING: in catalog but not in assets: {missing}")
    for extra in sorted(prepared - set(catalog)):
        print(f"WARNING: in assets but not in catalog: {extra}")
    if prepared == set(catalog):
        print(f"Catalog cross-check OK ({len(catalog)} objects).")
    (out_root / "catalog.json").write_text(
        json.dumps(catalog, indent=1, ensure_ascii=False) + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source", type=Path, default=None,
        help="Existing clone of elpis-lab/YCB_Dataset. "
             "If omitted, a shallow clone is made into a temp directory.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to place the prepared 'objects/ycb' tree in.",
    )
    args = parser.parse_args(argv)

    if args.source is not None:
        out_root = prepare(args.source.expanduser().resolve(), args.output_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="ycb_dataset_") as tmp:
            source = _clone_upstream(Path(tmp) / "YCB_Dataset")
            out_root = prepare(source, args.output_dir)

    print(
        "\nTo upload (requires write access to the org):\n"
        f"  hf upload UniEnvOrg/unienv_genesis_collection_assets \\\n"
        f"      {out_root} objects/ycb --repo-type dataset"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
