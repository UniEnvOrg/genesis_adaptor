"""The ``catalog.json`` writer (one copy — was four divergent variants).

The semantics that must never change (see the architecture review §1/§b
"already fine" list):

* **Merge** — incremental runs (``--objects``/category subsets into an
  already-populated tree) keep the entries of objects that are NOT part of
  this run; a re-run of the same object rewrites its entry (and its files),
  so a stale entry cannot outlive its packaged tree.
* **Drop** — the gate rewrite additionally passes ``drop`` with THIS run's
  gate-dropped ids so a failed re-run actually *removes* a previously
  published entry (it no longer loads correctly) instead of lingering,
  without clobbering the entries of objects this run never touched.

The family variances are parameters: ``overrides`` (each packager's
``CATALOG_OVERRIDES`` dict) and ``summary_label`` (artvip/lightwheel/molmo
count **articulated** entries in the log line; simready prints the literal
"rigid props" tag instead).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional, Sequence


def write_catalog(
    out_root: Path,
    records: Sequence[dict],
    *,
    merge: bool = True,
    drop: Optional[Sequence[str]] = None,
    overrides: Optional[Mapping[str, dict]] = None,
    summary_label: Optional[str] = "articulated",
) -> int:
    """Write ``catalog.json`` from the gate records (+ *overrides*).

    Entries: ``{"language_annotations": [...], "tags": [...]}`` (the shared
    ``load_catalog`` schema).  *summary_label* selects the log-line summary:
    ``"articulated"`` counts the entries tagged ``articulated``
    (artvip/lightwheel/molmospaces); any other truthy string is printed
    verbatim (simready's ``"rigid props"``); ``None`` prints just the count.

    Returns the number of entries written.
    """
    catalog_file = out_root / "catalog.json"
    catalog: dict = {}
    if merge and catalog_file.is_file():
        # Incremental runs (``--objects``/category subsets into an already-
        # populated tree) keep the entries of objects that are NOT part of
        # this run; a re-run of the same object overwrites its entry below.
        catalog = json.loads(catalog_file.read_text())

    for dropped_id in drop or ():
        catalog.pop(dropped_id, None)

    for record in records:
        entry = record["catalog_entry"]
        assert entry["language_annotations"] and entry["tags"] is not None
        catalog[record["object_id"]] = {
            "language_annotations": list(entry["language_annotations"]),
            "tags": sorted(set(entry["tags"])),
        }

    for object_id, override in (overrides or {}).items():
        if not override:
            continue
        base = catalog.get(object_id, {})
        override = dict(override)
        if "language_annotations" not in override:
            override["language_annotations"] = base.get("language_annotations")
        if "tags" not in override:
            override["tags"] = sorted(set(base.get("tags", [])) | set(override.get("add_tags", [])))
        override.pop("add_tags", None)
        catalog[object_id] = override

    # Iterate ``list(catalog)`` (artvip's variant): a re-run of the same
    # object rewrites its entry, so a stale entry cannot outlive its
    # packaged tree; iterating the live dict during the edit would not.
    for object_id in list(catalog):
        entry = catalog[object_id]
        if not entry.get("language_annotations") or entry.get("tags") is None:
            raise KeyError(
                f"Catalog entry for {object_id!r} is missing annotations/tags."
            )

    catalog_file.write_text(
        json.dumps(catalog, indent=1, ensure_ascii=False) + "\n"
    )
    if summary_label == "articulated":
        n_articulated = sum(1 for e in catalog.values() if "articulated" in e["tags"])
        print(
            f"Wrote {catalog_file} with {len(catalog)} entries "
            f"({n_articulated} articulated)."
        )
    elif summary_label:
        print(f"Wrote {catalog_file} with {len(catalog)} entries ({summary_label}).")
    else:
        print(f"Wrote {catalog_file} with {len(catalog)} entries.")
    return len(catalog)
