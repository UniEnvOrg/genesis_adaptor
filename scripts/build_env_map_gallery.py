#!/usr/bin/env python3
"""Generate a static HTML gallery for the env-map curation manifest.

Reads ``unienv_genesis_collection/assets/scenes/env_maps/info.json`` and
emits a single self-contained HTML file in the same ``env_maps/`` directory
(default: ``env_map_gallery.html``).  The gallery shows every HDRI's rendered
preview alongside three annotation layers: read-only VLM output, read-only
saved human overrides from ``info.json``, and editable local browser-side
overrides backed by ``localStorage``.

Design constraints
------------------

* **Dependency-free.**  Plain Python string templating; the stdlib is the only
  thing used at build time.  No Jinja2 / no Node / no build step.
* **Client-side filtering only.**  No external JS framework; a small inline
  ``<script>`` hides cards by ``data-`` attributes on a text search + status
  dropdown.  The whole page works offline.
* **Layered annotations.**  Because the gallery is commonly served via a static
  file server (or opened via ``file://``), saved manifest values remain
  read-only while browser-side overrides are edited locally and persisted to
  ``localStorage``.  They can be exported/imported as a JSON sidecar for later
  syncing back into ``info.json``.
* **Relative paths.**  Both the preview PNGs and the (optionally linked) HDR
  files are referenced relative to the HTML file's location, so the gallery
  can simply be opened in a browser with ``file://`` — no web server needed.
* **XSS-safe.**  All manifest strings (notes, status, paths, notes) are HTML
  escaped before being embedded in the template; the only HTML markup that
  survives is the one we generate ourselves.

The output is meant to be regenerated whenever the manifest changes, e.g.::

    python scripts/build_env_map_gallery.py

CLI flags allow pointing at a different manifest, env_maps directory, or
output path, but the defaults match the layout of this repository.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

# This script lives at ``genesis_adaptor/scripts/``; the collection's asset
# tree is one directory up under ``unienv_genesis_collection/assets/``.
_THIS_DIR = Path(__file__).resolve().parent
_COLLECTION_DIR = _THIS_DIR.parent / "unienv_genesis_collection"
_DEFAULT_ENV_MAP_DIR = _COLLECTION_DIR / "assets" / "scenes" / "env_maps"
_DEFAULT_MANIFEST = "info.json"
_DEFAULT_OUTPUT_NAME = "env_map_gallery.html"

# The set of status values surfaced in the filter dropdown.  ``"all"`` is
# appended automatically.  Order is preserved on screen.
_KNOWN_STATUSES = ("pending", "approved", "rejected", "failed")


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------
#
# The template is a single f-string.  Every interpolation that comes from
# user-editable manifest data is wrapped in :func:`html.escape` (see
# :func:`_render_cards`); everything else is a literal we control.  The page
# is intentionally plain (no external CSS/JS) so the file is portable.

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Env Map Curation Gallery</title>
<style>
  :root {{
    color-scheme: dark light;
    --bg: #11161d;
    --fg: #e6edf3;
    --muted: #8b949e;
    --card: #1b2230;
    --card-border: #2a3340;
    --accent: #58a6ff;
    --panel-neutral-bg: #f4f7fb;
    --panel-neutral-border: #cfd8e3;
    --panel-neutral-fg: #18212b;
    --panel-saved-bg: rgba(63, 185, 80, 0.12);
    --panel-saved-border: rgba(63, 185, 80, 0.55);
    --panel-saved-fg: #9be9a8;
    --panel-local-bg: rgba(248, 81, 73, 0.1);
    --panel-local-border: rgba(248, 81, 73, 0.55);
    --panel-local-fg: #ffb4af;
    --pending: #d29922;
    --approved: #3fb950;
    --rejected: #f85149;
    --failed: #db6d28;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    background: var(--bg);
    color: var(--fg);
    line-height: 1.4;
  }}
  header {{
    padding: 1.5rem 1.5rem 1rem;
    border-bottom: 1px solid var(--card-border);
    position: sticky;
    top: 0;
    background: var(--bg);
    z-index: 1;
  }}
  header h1 {{
    margin: 0 0 0.25rem;
    font-size: 1.4rem;
  }}
  header .meta {{
    color: var(--muted);
    font-size: 0.85rem;
    margin-bottom: 0.75rem;
  }}
  .controls {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    align-items: center;
  }}
  .controls input[type="search"],
  .controls select,
  .controls button {{
    background: var(--card);
    color: var(--fg);
    border: 1px solid var(--card-border);
    border-radius: 6px;
    padding: 0.4rem 0.6rem;
    font-size: 0.9rem;
  }}
  .controls input[type="search"] {{
    flex: 1 1 240px;
    min-width: 200px;
  }}
  .controls button {{
    cursor: pointer;
  }}
  .stats {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem 1rem;
    margin-top: 0.6rem;
    font-size: 0.85rem;
    color: var(--muted);
  }}
  .stats .badge {{
    display: inline-block;
    padding: 0.1rem 0.5rem;
    border-radius: 999px;
    background: var(--card);
    border: 1px solid var(--card-border);
  }}
  main {{
    padding: 1.5rem;
  }}
  .empty {{
    color: var(--muted);
    text-align: center;
    padding: 3rem 1rem;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 1rem;
  }}
  .card {{
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }}
  .card img {{
    width: 100%;
    aspect-ratio: 16 / 9;
    object-fit: cover;
    background: #000;
    display: block;
  }}
  .card .body {{
    padding: 0.7rem 0.85rem 0.85rem;
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }}
  .card h3 {{
    margin: 0;
    font-size: 0.95rem;
    word-break: break-word;
  }}
  .card h3 a {{
    color: var(--accent);
    text-decoration: none;
  }}
  .card h3 a:hover {{ text-decoration: underline; }}
  .row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.8rem;
    color: var(--muted);
  }}
  .status {{
    display: inline-block;
    padding: 0.1rem 0.55rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    border: 1px solid currentColor;
  }}
  .status.pending  {{ color: var(--pending); }}
  .status.approved {{ color: var(--approved); }}
  .status.rejected {{ color: var(--rejected); }}
  .status.failed   {{ color: var(--failed); }}
  .status.unknown  {{ color: var(--muted); }}
  .panel {{
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 0.65rem 0.75rem;
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }}
  .panel-title {{
    margin: 0;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    text-transform: uppercase;
  }}
  .panel-subtitle {{
    margin: -0.15rem 0 0;
    font-size: 0.76rem;
    color: var(--muted);
  }}
  .panel.vlm-panel {{
    background: var(--panel-neutral-bg);
    border-color: var(--panel-neutral-border);
    color: var(--panel-neutral-fg);
  }}
  .panel.vlm-panel .panel-title,
  .panel.vlm-panel .panel-value,
  .panel.vlm-panel .notes {{
    color: var(--panel-neutral-fg);
  }}
  .panel.vlm-panel .panel-subtitle {{
    color: rgba(24, 33, 43, 0.68);
  }}
  .panel.vlm-panel .panel-label {{
    color: rgba(24, 33, 43, 0.68);
  }}
  .panel.saved-panel {{
    background: var(--panel-saved-bg);
    border-color: var(--panel-saved-border);
  }}
  .panel.saved-panel .panel-title,
  .panel.saved-panel .panel-value,
  .panel.saved-panel .notes {{
    color: var(--panel-saved-fg);
  }}
  .panel.saved-panel .panel-subtitle,
  .panel.saved-panel .panel-label {{
    color: rgba(155, 233, 168, 0.82);
  }}
  .panel-row {{
    display: flex;
    justify-content: space-between;
    gap: 0.75rem;
    align-items: flex-start;
    font-size: 0.8rem;
  }}
  .panel-label {{
    color: var(--muted);
    flex: 0 0 auto;
  }}
  .panel-value {{
    text-align: right;
    word-break: break-word;
  }}
  .notes {{
    margin: 0.15rem 0 0;
    font-size: 0.85rem;
    color: var(--fg);
    white-space: pre-wrap;
    word-break: break-word;
  }}
  .notes.empty {{
    color: var(--muted);
    font-style: italic;
  }}
  .annotation-box {{
    margin-top: 0.45rem;
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
  }}
  .annotation-box.local-panel {{
    background: var(--panel-local-bg);
    border-color: var(--panel-local-border);
  }}
  .annotation-box.local-panel .panel-title,
  .annotation-box.local-panel .panel-value,
  .annotation-box.local-panel .local-layer-state {{
    color: var(--panel-local-fg);
  }}
  .annotation-box.local-panel .panel-subtitle,
  .annotation-box.local-panel .panel-label,
  .annotation-box.local-panel .annotation-hint {{
    color: rgba(255, 180, 175, 0.82);
  }}
  .annotation-grid {{
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 0.45rem;
    align-items: start;
  }}
  .local-summary {{
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 0.35rem 0.75rem;
    align-items: center;
  }}
  .local-layer-state {{
    font-size: 0.78rem;
  }}
  .annotation-box textarea,
  .annotation-box input[type="number"],
  .annotation-box button {{
    background: #121923;
    color: var(--fg);
    border: 1px solid var(--card-border);
    border-radius: 6px;
    font-size: 0.85rem;
  }}
  .annotation-box.local-panel textarea,
  .annotation-box.local-panel input[type="number"],
  .annotation-box.local-panel button {{
    border-color: rgba(248, 81, 73, 0.45);
  }}
  .annotation-box textarea {{
    width: 100%;
    min-height: 4.8rem;
    resize: vertical;
    padding: 0.45rem 0.55rem;
    font-family: inherit;
  }}
  .annotation-box input[type="number"] {{
    width: 100%;
    padding: 0.4rem 0.5rem;
  }}
  .annotation-buttons,
  .status-buttons {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
  }}
  .annotation-box button {{
    padding: 0.35rem 0.55rem;
    cursor: pointer;
  }}
  .annotation-box button:disabled {{
    opacity: 0.6;
    cursor: not-allowed;
  }}
  .status-buttons button.active {{
    border-color: var(--panel-local-border);
    box-shadow: inset 0 0 0 1px var(--panel-local-border);
  }}
  .annotation-hint {{
    color: var(--muted);
    font-size: 0.78rem;
  }}
  .last-error {{
    margin: 0.2rem 0 0;
    padding: 0.3rem 0.5rem;
    background: rgba(248, 81, 73, 0.12);
    border-left: 3px solid var(--rejected);
    border-radius: 4px;
    font-size: 0.8rem;
    color: #ffb4af;
    white-space: pre-wrap;
    word-break: break-word;
  }}
</style>
</head>
<body>
<header>
  <h1>Env Map Curation Gallery</h1>
  <div class="meta">
    Manifest: <code>{manifest_relpath}</code> &middot;
    Generated {generated_at} &middot;
    {entry_count} entr{entry_plural}
  </div>
  <div class="meta">
    White panels show read-only VLM output, green panels show saved manifest
    overrides, and red panels are your local browser-side edits. Export the red
    layer to JSON when you want to sync it back into <code>info.json</code>.
  </div>
  <div class="controls">
    <input type="search" id="search" placeholder="Search by HDR name, VLM/saved/local notes, or file&hellip;" autocomplete="off" spellcheck="false">
    <select id="status-filter" aria-label="Filter by status">
      <option value="all">All statuses</option>
      {status_options}
    </select>
    <button type="button" id="export-annotations">Export annotations</button>
    <button type="button" id="import-annotations">Import annotations</button>
    <button type="button" id="clear-annotations">Clear local annotations</button>
    <span id="visible-count" class="meta" aria-live="polite"></span>
  </div>
  <input type="file" id="import-file" accept="application/json" hidden>
  <div class="stats">{status_stats}</div>
</header>
<main>
  <div id="grid" class="grid">
{cards}
  </div>
  <p id="empty-state" class="empty" hidden>No env maps match the current filter.</p>
</main>
<script>
(function () {{
  var STORAGE_KEY = 'env_map_gallery_annotations_v1';
  var search = document.getElementById('search');
  var status = document.getElementById('status-filter');
  var grid = document.getElementById('grid');
  var emptyState = document.getElementById('empty-state');
  var visibleCount = document.getElementById('visible-count');
  var exportBtn = document.getElementById('export-annotations');
  var importBtn = document.getElementById('import-annotations');
  var clearBtn = document.getElementById('clear-annotations');
  var importFile = document.getElementById('import-file');
  var cards = Array.prototype.slice.call(grid.querySelectorAll('.card'));
  var annotations = loadAnnotations();

  function loadAnnotations() {{
    try {{
      var raw = window.localStorage.getItem(STORAGE_KEY);
      if (!raw) return {{}};
      var parsed = JSON.parse(raw);
      return parsed && typeof parsed === 'object' ? parsed : {{}};
    }} catch (_err) {{
      return {{}};
    }}
  }}

  function saveAnnotations() {{
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(annotations, null, 2));
  }}

  function getSavedState(card) {{
    return {{
      status: card.getAttribute('data-saved-status') || 'pending',
      rotation_deg: card.getAttribute('data-saved-rotation') || '0',
      notes: card.getAttribute('data-saved-notes') || '',
      hdr_path: card.getAttribute('data-hdr-path') || '',
      stem: card.getAttribute('data-stem') || '',
      static_search: card.getAttribute('data-static-search') || '',
    }};
  }}

  function getEffectiveState(card) {{
    var key = card.getAttribute('data-key');
    var saved = getSavedState(card);
    var local = annotations[key] || {{}};
    return {{
      status: local.status != null ? local.status : saved.status,
      rotation_deg: local.rotation_deg != null ? local.rotation_deg : saved.rotation_deg,
      notes: local.notes != null ? local.notes : saved.notes,
      hdr_path: saved.hdr_path,
      stem: saved.stem,
      static_search: saved.static_search,
      has_local_status: local.status != null,
      has_local_rotation: local.rotation_deg != null,
      has_local_notes: local.notes != null,
    }};
  }}

  function normaliseStatus(value) {{
    var token = String(value || '').trim().toLowerCase();
    if (!token) return 'pending';
    return token;
  }}

  function formatRotation(value) {{
    var rotNum = Number(value);
    if (Number.isFinite(rotNum)) {{
      return (Math.abs(rotNum - Math.round(rotNum)) < 1e-9
        ? String(Math.round(rotNum))
        : rotNum.toFixed(2)) + '\u00b0';
    }}
    return String(value || '0') + '\u00b0';
  }}

  function pruneAnnotation(key) {{
    if (!annotations[key]) return;
    if (annotations[key].status == null) delete annotations[key].status;
    if (annotations[key].notes == null) delete annotations[key].notes;
    if (annotations[key].rotation_deg == null) delete annotations[key].rotation_deg;
    if (Object.keys(annotations[key]).length === 0) delete annotations[key];
  }}

  function sameRotationValue(a, b) {{
    var aNum = Number(a);
    var bNum = Number(b);
    if (Number.isFinite(aNum) && Number.isFinite(bNum)) {{
      return Math.abs(aNum - bNum) < 1e-9;
    }}
    return String(a) === String(b);
  }}

  function normaliseStoredAnnotation(card) {{
    var key = card.getAttribute('data-key');
    var saved = getSavedState(card);
    var local = annotations[key];
    if (!local || typeof local !== 'object') return;
    if (local.status == null || normaliseStatus(local.status) === normaliseStatus(saved.status)) {{
      delete local.status;
    }}
    if (local.notes == null || local.notes === saved.notes) {{
      delete local.notes;
    }}
    if (local.rotation_deg == null || sameRotationValue(local.rotation_deg, saved.rotation_deg)) {{
      delete local.rotation_deg;
    }}
    pruneAnnotation(key);
  }}

  function updateCard(card) {{
    normaliseStoredAnnotation(card);
    var state = getEffectiveState(card);
    var statusToken = normaliseStatus(state.status);
    var notes = String(state.notes || '');
    var localStatusLabel = card.querySelector('.local-status-label');
    var localRotationLabel = card.querySelector('.local-rotation-label');
    var localLayerState = card.querySelector('.local-layer-state');
    var clearButton = card.querySelector('.clear-annotation');
    var noteInput = card.querySelector('.notes-input');
    var rotationInput = card.querySelector('.rotation-input');
    var searchBlob = [state.static_search, notes].join(' ').toLowerCase();
    var activeEdits = [];

    if (state.has_local_status) activeEdits.push('status');
    if (state.has_local_notes) activeEdits.push('notes');
    if (state.has_local_rotation) activeEdits.push('rotation');

    card.setAttribute('data-status', statusToken);
    card.setAttribute('data-search', searchBlob);

    localStatusLabel.textContent = statusToken;
    localStatusLabel.className = 'status local-status-label ' + (['pending','approved','rejected','failed'].indexOf(statusToken) >= 0 ? statusToken : 'unknown');
    localRotationLabel.textContent = formatRotation(state.rotation_deg);
    localLayerState.textContent = activeEdits.length
      ? ('Local overrides active: ' + activeEdits.join(', ') + '.')
      : 'No local overrides yet; inputs currently mirror the saved manifest values.';
    clearButton.disabled = activeEdits.length === 0;

    if (document.activeElement !== noteInput) noteInput.value = notes;
    if (document.activeElement !== rotationInput) rotationInput.value = state.rotation_deg;

    Array.prototype.forEach.call(card.querySelectorAll('.status-setter'), function (btn) {{
      btn.classList.toggle('active', btn.getAttribute('data-status-value') === statusToken);
    }});
  }}

  function updateAllCards() {{
    cards.forEach(updateCard);
    applyFilters();
  }}

  function applyFilters() {{
    var q = search.value.trim().toLowerCase();
    var s = status.value;
    var shown = 0;
    for (var i = 0; i < cards.length; i++) {{
      var c = cards[i];
      var matchesStatus = (s === 'all') || (c.getAttribute('data-status') === s);
      var matchesSearch = !q || (c.getAttribute('data-search') || '').indexOf(q) !== -1;
      var visible = matchesStatus && matchesSearch;
      c.hidden = !visible;
      if (visible) {{ shown += 1; }}
    }}
    emptyState.hidden = shown !== 0;
    visibleCount.textContent = shown === cards.length
      ? ''
      : (shown + ' of ' + cards.length + ' shown');
  }}

  function setStatus(card, value) {{
    var key = card.getAttribute('data-key');
    var saved = getSavedState(card);
    annotations[key] = annotations[key] || {{}};
    annotations[key].status = value === normaliseStatus(saved.status) ? null : value;
    normaliseStoredAnnotation(card);
    saveAnnotations();
    updateCard(card);
    applyFilters();
  }}

  function saveCardEdits(card) {{
    var key = card.getAttribute('data-key');
    var saved = getSavedState(card);
    var notesValue = card.querySelector('.notes-input').value;
    var rotationValue = card.querySelector('.rotation-input').value;
    annotations[key] = annotations[key] || {{}};
    annotations[key].notes = notesValue === saved.notes ? null : notesValue;
    annotations[key].rotation_deg = sameRotationValue(rotationValue, saved.rotation_deg) ? null : rotationValue;
    normaliseStoredAnnotation(card);
    saveAnnotations();
    updateCard(card);
    applyFilters();
  }}

  function clearCardEdits(card) {{
    var key = card.getAttribute('data-key');
    delete annotations[key];
    saveAnnotations();
    updateCard(card);
    applyFilters();
  }}

  function downloadAnnotations() {{
    var payload = {{
      exported_at: new Date().toISOString(),
      storage_key: STORAGE_KEY,
      annotations: annotations,
    }};
    var blob = new Blob([JSON.stringify(payload, null, 2)], {{ type: 'application/json' }});
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = 'env_map_gallery_annotations.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }}

  function importAnnotationsFromFile(file) {{
    var reader = new FileReader();
    reader.onload = function () {{
      try {{
        var parsed = JSON.parse(String(reader.result || '{{}}'));
        var incoming = parsed.annotations || parsed;
        if (!incoming || typeof incoming !== 'object') throw new Error('Invalid annotations JSON');
        Object.keys(incoming).forEach(function (key) {{
          if (!incoming[key] || typeof incoming[key] !== 'object') return;
          annotations[key] = Object.assign(annotations[key] || {{}}, incoming[key]);
        }});
        cards.forEach(normaliseStoredAnnotation);
        saveAnnotations();
        updateAllCards();
      }} catch (err) {{
        window.alert('Failed to import annotations: ' + err);
      }}
    }};
    reader.readAsText(file);
  }}

  function clearAllAnnotations() {{
    if (!window.confirm('Clear all local browser annotations for this gallery?')) return;
    annotations = {{}};
    saveAnnotations();
    updateAllCards();
  }}

  cards.forEach(function (card) {{
    Array.prototype.forEach.call(card.querySelectorAll('.status-setter'), function (btn) {{
      btn.addEventListener('click', function () {{
        setStatus(card, btn.getAttribute('data-status-value'));
      }});
    }});
    card.querySelector('.notes-input').addEventListener('input', function () {{
      saveCardEdits(card);
    }});
    card.querySelector('.rotation-input').addEventListener('input', function () {{
      saveCardEdits(card);
    }});
    card.querySelector('.save-annotation').addEventListener('click', function () {{
      saveCardEdits(card);
    }});
    card.querySelector('.clear-annotation').addEventListener('click', function () {{
      clearCardEdits(card);
    }});
  }});

  exportBtn.addEventListener('click', downloadAnnotations);
  importBtn.addEventListener('click', function () {{
    importFile.click();
  }});
  importFile.addEventListener('change', function () {{
    if (importFile.files && importFile.files[0]) {{
      importAnnotationsFromFile(importFile.files[0]);
    }}
    importFile.value = '';
  }});
  clearBtn.addEventListener('click', clearAllAnnotations);

  search.addEventListener('input', applyFilters);
  status.addEventListener('change', applyFilters);
  cards.forEach(normaliseStoredAnnotation);
  saveAnnotations();
  updateAllCards();
}})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """UTC timestamp in the same ``Z``-suffixed ISO-8601 form the manifest uses."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_rotation(rotation: object) -> str:
    """Render the user-editable rotation hint, stripping a trailing ``.0``."""
    try:
        value = float(rotation)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return html.escape(str(rotation))
    if value == int(value):
        return f"{int(value)}\u00b0"
    return f"{value:.2f}\u00b0"


def _format_rendered_at(ts: object) -> str:
    """Pretty-print an ISO-8601 timestamp; fall back to the raw value on errors."""
    if not isinstance(ts, str) or not ts:
        return html.escape(str(ts)) if ts else ""
    try:
        # Manifest uses ``...Z``; ``fromisoformat`` accepts that on Py 3.11+.
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return html.escape(ts)


def _normalise_status(raw: object) -> str:
    """Map a status string to a CSS class token; default to ``unknown``."""
    if not isinstance(raw, str) or not raw:
        return "unknown"
    token = raw.strip().lower()
    return token if token in _KNOWN_STATUSES else "unknown"


def _status_label(raw: object) -> str:
    """Display label for a status (capitalised, falls back to the raw token)."""
    token = _normalise_status(raw)
    if token == "unknown" and isinstance(raw, str) and raw.strip():
        return raw.strip()
    return token or "unknown"


def _display_value(raw: object, *, empty: str = "—") -> str:
    """Escape a small scalar value, falling back to ``empty`` when blank."""
    if raw is None:
        return empty
    text = str(raw).strip()
    return html.escape(text) if text else empty


def _format_optional_rotation(rotation: object) -> str:
    """Format an optional rotation value, returning an em dash when blank."""
    if rotation in (None, ""):
        return "—"
    return _format_rotation(rotation)


def _render_panel_rows(rows: List[tuple[str, str]]) -> str:
    """Render a set of read-only key/value rows for a card sub-panel."""
    return "\n".join(
        f'        <div class="panel-row"><span class="panel-label">{html.escape(label)}</span>'
        f'<span class="panel-value">{value}</span></div>'
        for label, value in rows
    )


def _render_panel_notes(notes_raw: object, *, empty_label: str) -> str:
    """Render a notes block for a panel."""
    notes_text = str(notes_raw or "")
    if notes_text.strip():
        return f'<p class="notes">{html.escape(notes_text)}</p>'
    return f'<p class="notes empty">{html.escape(empty_label)}</p>'


def _build_card(stem: str, entry: dict) -> Optional[str]:
    """Render one entry as a card, or ``None`` if it has no preview image."""
    if not isinstance(entry, dict):
        return None
    preview_rel = entry.get("preview_path")
    if not isinstance(preview_rel, str) or not preview_rel:
        return None

    # ``preview_path`` is relative to the env_maps directory; the HTML lives
    # next to the previews/ folder, so the path can be used verbatim.
    preview_rel = preview_rel.replace("\\", "/")

    hdr_rel = entry.get("hdr_path", "")
    if not isinstance(hdr_rel, str):
        hdr_rel = ""
    hdr_rel = hdr_rel.replace("\\", "/")
    hdr_href = html.escape(hdr_rel) if hdr_rel else ""

    vlm = entry.get("vlm") if isinstance(entry.get("vlm"), dict) else None
    saved_status_value = entry.get("status") or "pending"
    saved_rotation_value = entry.get("rotation_deg")
    if saved_rotation_value in (None, ""):
        saved_rotation_value = 0.0
    saved_notes_raw = entry.get("notes") or ""

    status_token = _normalise_status(saved_status_value)
    status_lbl = html.escape(_status_label(saved_status_value))
    rendered = _format_rendered_at(entry.get("rendered_at"))
    last_error = entry.get("_last_error") or ""

    # The ``data-search`` blob is what the client-side filter matches against.
    # We fold in the HDR stem (which is what users will most often type), the
    # ``hdr_path``, and the notes so any of them can be used to find an entry.
    search_blob = " ".join(
        str(x)
        for x in (
            stem,
            hdr_rel,
            saved_notes_raw,
            (vlm or {}).get("context_type", "") if vlm is not None else "",
            (vlm or {}).get("decision", "") if vlm is not None else "",
            (vlm or {}).get("notes", "") if vlm is not None else "",
        )
        if x
    ).lower()

    name_html = (
        f'<a href="{hdr_href}">{html.escape(stem)}</a>'
        if hdr_href
        else html.escape(stem)
    )

    error_html = ""
    if last_error:
        error_html = (
            f'<p class="last-error">'
            f'<strong>Last render error:</strong> {html.escape(last_error)}'
            f'</p>'
        )

    rendered_html = (
        f'<div class="row"><span>Rendered</span><span>{rendered}</span></div>'
        if rendered
        else ""
    )

    vlm_panel_html = ""
    if vlm is not None:
        vlm_rows = [
            ("Decision", _display_value(vlm.get("decision"))),
            ("Space score", _display_value(vlm.get("space_score"))),
            ("Context", _display_value(vlm.get("context_type"))),
        ]
        if vlm.get("rotation_hint_deg") not in (None, ""):
            vlm_rows.append(
                ("Rotation hint", _format_optional_rotation(vlm.get("rotation_hint_deg")))
            )
        vlm_panel_html = (
            '      <section class="panel vlm-panel">\n'
            '        <h4 class="panel-title">VLM output</h4>\n'
            '        <p class="panel-subtitle">Read-only model annotation for comparison.</p>\n'
            f'{_render_panel_rows(vlm_rows)}\n'
            f'        {_render_panel_notes(vlm.get("notes"), empty_label="no VLM notes")}\n'
            '      </section>\n'
        )

    saved_panel_html = (
        '      <section class="panel saved-panel">\n'
        '        <h4 class="panel-title">Saved human override</h4>\n'
        '        <p class="panel-subtitle">Read-only values currently stored in info.json.</p>\n'
        f'{_render_panel_rows([("Status", status_lbl), ("Rotation", _format_optional_rotation(saved_rotation_value))])}\n'
        f'        {_render_panel_notes(saved_notes_raw, empty_label="no saved notes")}\n'
        '      </section>\n'
    )

    notes_attr = html.escape(str(saved_notes_raw), quote=True)
    rotation_attr = html.escape(str(saved_rotation_value), quote=True)

    return (
        f'  <article class="card" '
        f'data-key="{html.escape(stem, quote=True)}" '
        f'data-stem="{html.escape(stem, quote=True)}" '
        f'data-hdr-path="{html.escape(hdr_rel, quote=True)}" '
        f'data-saved-status="{html.escape(str(saved_status_value), quote=True)}" '
        f'data-saved-rotation="{rotation_attr}" '
        f'data-saved-notes="{notes_attr}" '
        f'data-status="{status_token}" '
        f'data-static-search="{html.escape(search_blob, quote=True)}" '
        f'data-search="{html.escape(search_blob, quote=True)}">\n'
        f'    <img src="{html.escape(preview_rel, quote=True)}" '
        f'loading="lazy" alt="Preview of {html.escape(stem)}">\n'
        f'    <div class="body">\n'
        f'      <h3>{name_html}</h3>\n'
        f'      {rendered_html}\n'
        f'      {error_html}\n'
        f'{vlm_panel_html}'
        f'{saved_panel_html}'
        f'      <div class="panel annotation-box local-panel">\n'
        f'        <h4 class="panel-title">Local browser edits</h4>\n'
        f'        <p class="panel-subtitle">Editable red layer used for local filtering/export only.</p>\n'
        f'        <div class="local-summary">\n'
        f'          <span class="panel-label">Effective status</span>\n'
        f'          <span class="status local-status-label {status_token}">{status_lbl}</span>\n'
        f'          <span class="panel-label">Effective rotation</span>\n'
        f'          <span class="panel-value local-rotation-label">{_format_optional_rotation(saved_rotation_value)}</span>\n'
        f'        </div>\n'
        f'        <div class="local-layer-state">No local overrides yet; inputs currently mirror the saved manifest values.</div>\n'
        f'        <div class="status-buttons">\n'
        f'          <button type="button" class="status-setter" data-status-value="pending">Pending</button>\n'
        f'          <button type="button" class="status-setter" data-status-value="approved">Approve</button>\n'
        f'          <button type="button" class="status-setter" data-status-value="rejected">Reject</button>\n'
        f'          <button type="button" class="status-setter" data-status-value="failed">Failed</button>\n'
        f'        </div>\n'
        f'        <div class="annotation-grid">\n'
        f'          <textarea class="notes-input" placeholder="Add local browser-only notes for this HDRI...">{html.escape(str(saved_notes_raw))}</textarea>\n'
        f'          <input class="rotation-input" type="number" step="1" value="{html.escape(str(saved_rotation_value), quote=True)}" aria-label="Local rotation degrees">\n'
        f'        </div>\n'
        f'        <div class="annotation-buttons">\n'
        f'          <button type="button" class="save-annotation">Save local annotation</button>\n'
        f'          <button type="button" class="clear-annotation">Reset local annotation</button>\n'
        f'        </div>\n'
        f'        <div class="annotation-hint">Edits auto-save to your browser via localStorage and are never written back to the manifest automatically.</div>\n'
        f'      </div>\n'
        f'    </div>\n'
        f'  </article>'
    )


def _render_cards(entries: Dict[str, dict]) -> tuple[List[str], Dict[str, int]]:
    """Render manifest entries as card markup and a status counter."""
    cards: List[str] = []
    counts: Dict[str, int] = {s: 0 for s in _KNOWN_STATUSES}
    counts["unknown"] = 0
    # Sort alphabetically by stem for a stable, predictable layout.
    for stem in sorted(entries.keys()):
        entry = entries[stem]
        token = _normalise_status(entry.get("status")) if isinstance(entry, dict) else "unknown"
        counts[token] = counts.get(token, 0) + 1
        card = _build_card(stem, entry)
        if card is not None:
            cards.append(card)
    return cards, counts


def _build_status_options(counts: Dict[str, int]) -> str:
    """Build the ``<option>`` list for the status filter dropdown."""
    options = []
    # Show the canonical statuses first, then any other tokens we observed.
    seen = set()
    for status in list(_KNOWN_STATUSES) + sorted(
        k for k in counts if k not in _KNOWN_STATUSES
    ):
        if status in seen:
            continue
        seen.add(status)
        n = counts.get(status, 0)
        label = status if status != "unknown" else "other"
        options.append(
            f'<option value="{html.escape(status)}">'
            f'{html.escape(label)} ({n})</option>'
        )
    return "\n      ".join(options)


def _build_status_stats(counts: Dict[str, int], total: int) -> str:
    """Build the small status counters shown in the header."""
    badges = []
    for status in list(_KNOWN_STATUSES) + sorted(
        k for k in counts if k not in _KNOWN_STATUSES
    ):
        n = counts.get(status, 0)
        if n == 0:
            continue
        label = status if status != "unknown" else "other"
        badges.append(
            f'<span class="badge">'
            f'{html.escape(label)}: {n}/{total}'
            f'</span>'
        )
    return "\n    ".join(badges)


def _load_manifest(manifest_path: Path) -> dict:
    """Load ``info.json`` from disk, returning an empty skeleton on failure."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"Manifest root must be a JSON object, got {type(data).__name__}"
        )
    entries = data.get("entries")
    if not isinstance(entries, dict):
        entries = {}
    data["entries"] = entries
    return data


def _resolve_paths(
    env_map_dir_arg: Optional[Path],
    manifest_arg: Optional[Path],
    output_arg: Optional[Path],
) -> tuple[Path, Path, Path]:
    """Apply defaults and resolve the env_map/manifest/output paths."""
    env_map_dir = (env_map_dir_arg or _DEFAULT_ENV_MAP_DIR).resolve()
    manifest_path = (manifest_arg or env_map_dir / _DEFAULT_MANIFEST).resolve()
    output_path = (output_arg or env_map_dir / _DEFAULT_OUTPUT_NAME).resolve()
    return env_map_dir, manifest_path, output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build a static HTML gallery for the env-map curation manifest. "
            "Reads <env-map-dir>/info.json and writes "
            "<env-map-dir>/env_map_gallery.html by default."
        ),
    )
    p.add_argument(
        "--env-map-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing the .hdr files, previews/ subfolder, and "
            "info.json. "
            f"Default: {_DEFAULT_ENV_MAP_DIR}"
        ),
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Path to the curation manifest. "
            "Default: <env-map-dir>/info.json"
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path to the generated HTML file. "
            "Default: <env-map-dir>/env_map_gallery.html"
        ),
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the per-run summary line.",
    )
    return p


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    env_map_dir, manifest_path, output_path = _resolve_paths(
        args.env_map_dir, args.manifest, args.output
    )

    if not env_map_dir.is_dir():
        print(
            f"ERROR: env-map directory not found: {env_map_dir}",
            file=sys.stderr,
        )
        return 2
    if not manifest_path.is_file():
        print(
            f"ERROR: manifest not found: {manifest_path}",
            file=sys.stderr,
        )
        return 2

    manifest = _load_manifest(manifest_path)
    entries = manifest.get("entries", {})
    cards, counts = _render_cards(entries)
    total = len(entries)

    if not cards:
        print(
            "WARNING: manifest contains no renderable entries (no preview_path).",
            file=sys.stderr,
        )

    # ``manifest_relpath`` is what the user will see in the header; make it
    # relative to the env_maps dir so it stays short even when the script
    # is run from elsewhere.
    try:
        manifest_relpath = manifest_path.relative_to(env_map_dir).as_posix()
    except ValueError:
        manifest_relpath = manifest_path.name

    html_doc = _HTML_TEMPLATE.format(
        manifest_relpath=html.escape(manifest_relpath),
        generated_at=html.escape(_now_iso()),
        entry_count=total,
        entry_plural="y" if total == 1 else "ies",
        status_options=_build_status_options(counts),
        status_stats=_build_status_stats(counts, total),
        cards="\n".join(cards) if cards else '    <p class="empty">No env maps to show.</p>',
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_doc, encoding="utf-8")

    if not args.quiet:
        print(
            f"Wrote {output_path}  "
            f"({total} entr{'y' if total == 1 else 'ies'}, "
            f"{len(cards)} card{'s' if len(cards) != 1 else ''})"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
