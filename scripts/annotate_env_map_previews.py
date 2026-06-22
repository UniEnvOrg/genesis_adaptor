#!/usr/bin/env python3
"""Annotate env-map preview images with an Anthropic-compatible VLM.

Reads preview PNGs from
``unienv_genesis_collection/assets/scenes/env_maps/previews`` and updates the
existing sibling manifest ``info.json`` in-place by adding a nested
``vlm`` object to each annotated entry.

Only stdlib modules are used for HTTP, JSON, and base64 handling.

Examples
--------

Dry-run the first five matching entries::

    python scripts/annotate_env_map_previews.py --limit 5 --dry-run

Annotate only preview names containing "studio"::

    python scripts/annotate_env_map_previews.py --pattern "*studio*"

Override endpoint settings via CLI or environment::

    ANTHROPIC_BASE_URL=https://api.anthropic.com \
    ANTHROPIC_API_KEY=... \
    ANTHROPIC_MODEL=claude-3-5-sonnet-latest \
    python scripts/annotate_env_map_previews.py --limit 10
"""

from __future__ import annotations

import argparse
import base64
import fnmatch
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


_THIS_DIR = Path(__file__).resolve().parent
_COLLECTION_DIR = _THIS_DIR.parent / "unienv_genesis_collection"
_DEFAULT_ENV_MAP_DIR = _COLLECTION_DIR / "assets" / "scenes" / "env_maps"
_DEFAULT_PREVIEW_DIR = _DEFAULT_ENV_MAP_DIR / "previews"
_DEFAULT_MANIFEST_PATH = _DEFAULT_ENV_MAP_DIR / "info.json"

_ENV_BASE_URL = "ANTHROPIC_BASE_URL"
_ENV_API_KEY = "ANTHROPIC_API_KEY"
_ENV_MODEL = "ANTHROPIC_MODEL"
_ENV_LIMIT = "ENV_MAP_VLM_LIMIT"
_ENV_PATTERN = "ENV_MAP_VLM_PATTERN"
_ENV_FORCE = "ENV_MAP_VLM_FORCE"

_DEFAULT_BASE_URL = "https://api.anthropic.com"
_DEFAULT_MODEL = "claude-3-5-sonnet-latest"
_ANTHROPIC_VERSION = "2023-06-01"
_DEFAULT_TIMEOUT_SEC = 120
_DEFAULT_RETRIES = 3
_DEFAULT_MAX_TOKENS = 320
_UA = (
    "curl/8.5.0"
)

_PROMPT = """You are reviewing an HDRI preview for tabletop robot scenes.

Judge whether this environment map is suitable for a tabletop robot setup.
Focus especially on spatial openness: does the scene feel spacious enough
around the robot and table, or does it feel cramped, enclosed, cluttered,
narrow, close-up, or lacking visible surrounding space? Also consider general
contextual suitability for robot tabletop scenes.

Return exactly one JSON object with this schema:
{
  "decision": "approve" | "reject" | "maybe",
  "space_score": number,
  "context_type": string,
  "notes": string,
  "rotation_hint_deg": number or null,
  "confidence": number
}

Rules:
- `space_score` must be from 0 to 10, where 10 means very open/spacious.
- `context_type` should be a short snake_case label like `studio`,
  `open_indoor`, `cramped_indoor`, `warehouse`, `outdoor`, `industrial`,
  `urban`, `nature`, or `other`.
- `notes` should briefly mention openness/crampedness and overall suitability.
- `rotation_hint_deg` is optional; use null if no strong suggestion.
- `confidence` must be from 0 to 1.
- Output JSON only. No markdown, no code fences, no extra prose.
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer.") from exc


def _load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("Manifest root must be a JSON object.")
    entries = manifest.get("entries")
    if not isinstance(entries, dict):
        raise ValueError("Manifest must contain an object field 'entries'.")
    return manifest


def _save_manifest_atomic(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    tmp_path.replace(path)


def _messages_url(base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/v1/messages"):
        return trimmed
    if trimmed.endswith("/v1"):
        return trimmed + "/messages"
    return trimmed + "/v1/messages"


def _matches_pattern(stem: str, entry: dict, pattern: Optional[str]) -> bool:
    if not pattern:
        return True
    pattern_lc = pattern.lower()
    haystacks = [
        stem,
        str(entry.get("hdr_path", "")),
        str(entry.get("preview_path", "")),
    ]
    return any(fnmatch.fnmatchcase(text.lower(), pattern_lc) for text in haystacks)


def _resolve_preview_path(preview_dir: Path, env_map_dir: Path, stem: str, entry: dict) -> Optional[Path]:
    preview_rel = entry.get("preview_path")
    if isinstance(preview_rel, str) and preview_rel:
        candidate = env_map_dir / preview_rel
        if candidate.is_file():
            return candidate
    fallback = preview_dir / f"{stem}.png"
    if fallback.is_file():
        return fallback
    return None


def _iter_candidates(
    entries: Dict[str, dict],
    *,
    preview_dir: Path,
    env_map_dir: Path,
    pattern: Optional[str],
    force: bool,
) -> List[Tuple[str, dict, Path]]:
    candidates: List[Tuple[str, dict, Path]] = []
    for stem in sorted(entries):
        entry = entries[stem]
        if not isinstance(entry, dict):
            continue
        if not _matches_pattern(stem, entry, pattern):
            continue
        preview_path = _resolve_preview_path(preview_dir, env_map_dir, stem, entry)
        if preview_path is None:
            continue
        if not force and isinstance(entry.get("vlm"), dict):
            continue
        candidates.append((stem, entry, preview_path))
    return candidates


def _read_image_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _post_messages(
    *,
    url: str,
    api_key: str,
    model: str,
    image_b64: str,
    timeout_sec: int,
    retries: int,
) -> dict:
    payload = {
        "model": model,
        "max_tokens": _DEFAULT_MAX_TOKENS,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": _PROMPT},
                ],
            }
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "content-type": "application/json",
        "x-api-key": api_key,
        "authorization": f"Bearer {api_key}",
        "anthropic-version": _ANTHROPIC_VERSION,
        "user-agent": _UA,
        "accept": "application/json",
    }

    last_error: Optional[BaseException] = None
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as response:  # noqa: S310
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            retryable = exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
            last_error = RuntimeError(
                f"HTTP {exc.code} from VLM endpoint: {body.strip() or exc.reason}"
            )
            if not retryable or attempt >= retries:
                raise last_error
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= retries:
                raise
        time.sleep(min(2 ** (attempt - 1), 8))
    if last_error is None:
        raise RuntimeError("Unknown VLM request failure.")
    raise last_error


def _extract_response_text(response: dict) -> str:
    content = response.get("content")
    if isinstance(content, list):
        text_parts: List[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        if text_parts:
            return "\n".join(text_parts).strip()
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    raise ValueError("Response did not contain any text content.")


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_json_object(text: str) -> dict:
    decoder = json.JSONDecoder()
    candidates = [text.strip(), _strip_code_fences(text)]
    seen = set()
    for candidate in candidates:
        if candidate in seen or not candidate:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
        for idx, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                parsed, _end = decoder.raw_decode(candidate[idx:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    raise ValueError("Could not extract a JSON object from model output.")


def _coerce_float(value: object, field_name: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        return float(value)
    raise ValueError(f"Field {field_name!r} must be numeric.")


def _normalize_decision(value: object) -> str:
    if isinstance(value, bool):
        return "approve" if value else "reject"
    decision = str(value).strip().lower()
    mapping = {
        "approve": "approve",
        "approved": "approve",
        "accept": "approve",
        "accepted": "approve",
        "suitable": "approve",
        "yes": "approve",
        "reject": "reject",
        "rejected": "reject",
        "unsuitable": "reject",
        "no": "reject",
        "maybe": "maybe",
        "borderline": "maybe",
        "unclear": "maybe",
        "unsure": "maybe",
    }
    if decision in mapping:
        return mapping[decision]
    raise ValueError(f"Unrecognized decision value: {value!r}")


def _normalize_context_type(value: object) -> str:
    context_type = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return context_type or "other"


def _normalize_annotation(raw: dict, *, model: str, annotated_at: str) -> dict:
    decision = _normalize_decision(raw.get("decision"))
    space_score = _coerce_float(raw.get("space_score"), "space_score")
    confidence = _coerce_float(raw.get("confidence"), "confidence")
    if not 0.0 <= space_score <= 10.0:
        raise ValueError(f"space_score out of range: {space_score}")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence out of range: {confidence}")

    notes = raw.get("notes", "")
    if notes is None:
        notes = ""
    notes = str(notes).strip()

    annotation = {
        "model": model,
        "annotated_at": annotated_at,
        "decision": decision,
        "space_score": space_score,
        "context_type": _normalize_context_type(raw.get("context_type", "other")),
        "notes": notes,
        "confidence": confidence,
    }

    rotation_hint = raw.get("rotation_hint_deg")
    if rotation_hint not in (None, ""):
        annotation["rotation_hint_deg"] = _coerce_float(
            rotation_hint,
            "rotation_hint_deg",
        )
    return annotation


def _request_annotation(
    *,
    url: str,
    api_key: str,
    model: str,
    preview_path: Path,
    timeout_sec: int,
    retries: int,
) -> dict:
    response = _post_messages(
        url=url,
        api_key=api_key,
        model=model,
        image_b64=_read_image_b64(preview_path),
        timeout_sec=timeout_sec,
        retries=retries,
    )
    response_model = response.get("model")
    normalized_model = str(response_model).strip() if response_model else model
    text = _extract_response_text(response)
    raw = _extract_json_object(text)
    return _normalize_annotation(raw, model=normalized_model, annotated_at=_now_iso())


def _build_parser() -> argparse.ArgumentParser:
    env_limit = _env_int(_ENV_LIMIT)
    parser = argparse.ArgumentParser(
        description="Annotate env-map preview PNGs with an Anthropic-compatible VLM.",
    )
    parser.set_defaults(force=_env_bool(_ENV_FORCE, default=False))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_DEFAULT_MANIFEST_PATH,
        help="Path to info.json.",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=_DEFAULT_PREVIEW_DIR,
        help="Directory containing preview PNGs.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get(_ENV_BASE_URL, _DEFAULT_BASE_URL),
        help=f"Anthropic-compatible API base URL. Env: {_ENV_BASE_URL}.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get(_ENV_API_KEY),
        help=f"API key for the endpoint. Env: {_ENV_API_KEY}.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get(_ENV_MODEL, _DEFAULT_MODEL),
        help=f"Model name for /v1/messages. Env: {_ENV_MODEL}.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=env_limit,
        help=f"Maximum number of entries to annotate. Env: {_ENV_LIMIT}.",
    )
    parser.add_argument(
        "--pattern",
        default=os.environ.get(_ENV_PATTERN),
        help=f"Case-insensitive glob filter for stem/hdr/preview paths. Env: {_ENV_PATTERN}.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=f"Re-annotate entries that already have a 'vlm' object. Env: {_ENV_FORCE}=1.",
    )
    parser.add_argument(
        "--no-force",
        dest="force",
        action="store_false",
        help="Disable forced re-annotation even if ENV_MAP_VLM_FORCE is set.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching entries without calling the VLM API or writing the manifest.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=_DEFAULT_TIMEOUT_SEC,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=_DEFAULT_RETRIES,
        help="Number of attempts per annotation request.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive when provided.")
    if args.timeout <= 0:
        parser.error("--timeout must be positive.")
    if args.retries <= 0:
        parser.error("--retries must be positive.")

    manifest_path = args.manifest.resolve()
    preview_dir = args.preview_dir.resolve()
    env_map_dir = manifest_path.parent

    if not manifest_path.is_file():
        parser.error(f"Manifest not found: {manifest_path}")
    if not preview_dir.is_dir():
        parser.error(f"Preview directory not found: {preview_dir}")

    manifest = _load_manifest(manifest_path)
    entries = manifest["entries"]
    candidates = _iter_candidates(
        entries,
        preview_dir=preview_dir,
        env_map_dir=env_map_dir,
        pattern=args.pattern,
        force=args.force,
    )
    if args.limit is not None:
        candidates = candidates[: args.limit]

    print(f"Manifest: {manifest_path}")
    print(f"Preview dir: {preview_dir}")
    print(f"Endpoint: {_messages_url(args.base_url)}")
    print(f"Model: {args.model}")
    print(f"Selected entries: {len(candidates)}")

    if args.dry_run:
        for stem, _entry, preview_path in candidates:
            print(f"DRY RUN  {stem} <- {preview_path.name}")
        return 0

    if not candidates:
        print("Nothing to annotate.")
        return 0
    if not args.api_key:
        parser.error(
            f"An API key is required via --api-key or environment variable {_ENV_API_KEY}."
        )

    url = _messages_url(args.base_url)
    successes = 0
    failures = 0

    for index, (stem, entry, preview_path) in enumerate(candidates, start=1):
        print(f"[{index}/{len(candidates)}] Annotating {stem} ...")
        try:
            annotation = _request_annotation(
                url=url,
                api_key=args.api_key,
                model=args.model,
                preview_path=preview_path,
                timeout_sec=args.timeout,
                retries=args.retries,
            )
            entry["vlm"] = annotation
            _save_manifest_atomic(manifest_path, manifest)
            successes += 1
            print(
                f"  -> {annotation['decision']} "
                f"space={annotation['space_score']:.1f} "
                f"context={annotation['context_type']}"
            )
        except Exception as exc:
            failures += 1
            print(f"  ERROR: {exc}", file=sys.stderr)

    print(f"Done. successes={successes} failures={failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
