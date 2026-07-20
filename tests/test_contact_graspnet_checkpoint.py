"""Deterministic tests for Contact-GraspNet checkpoint download hardening.

All network access is mocked; the real ~26MB checkpoint is never fetched.
The expected size/SHA-256 class constants are monkeypatched to match a small
synthetic payload so integrity validation can be exercised end to end.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Genesis must be initialized before importing unienv_genesis modules (their
# transitive genesis imports require an initialized backend at module level).
import genesis as gs
from genesis.constants import backend as gs_backend

if not gs._initialized:
    gs.init(backend=gs_backend.cpu)

from unienv_genesis_refpolicy.grasp_providers.contact_graspnet import (
    ContactGraspNetGraspCandidateProvider,
)

Provider = ContactGraspNetGraspCandidateProvider

PINNED_COMMIT = "a3c92388a5134ac0ad6d200d6e51b6317656cc3f"
PINNED_SHA256 = "39fc3439d5814043ba64e0715c127c2e8aca6ea376c22614e11af1bc89317762"
PINNED_SIZE = 26_628_146

PAYLOAD = (b"fake-contact-graspnet-checkpoint\x00" * 4096) + b"end"
PAYLOAD_SHA256 = hashlib.sha256(PAYLOAD).hexdigest()


class _FakeResponse:
    """A minimal file-like urlopen response with optional mid-stream failure."""

    def __init__(self, data: bytes, *, fail_on_read: int | None = None) -> None:
        self._data = data
        self._offset = 0
        self._reads = 0
        self._fail_on_read = fail_on_read

    def read(self, size: int = -1) -> bytes:
        self._reads += 1
        if self._fail_on_read is not None and self._reads > self._fail_on_read:
            raise ConnectionError("simulated network interruption")
        if self._offset >= len(self._data):
            return b""
        if size is None or size < 0:
            size = len(self._data) - self._offset
        chunk = self._data[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc) -> bool:
        return False


def _make_provider(checkpoint_dir: Path) -> Provider:
    """Build a real provider; only Genesis-heavy collaborators are mocked."""
    world = MagicMock()
    world.batch_size = None
    return Provider(
        world=world,
        camera_node=MagicMock(),
        target_entity=MagicMock(),
        checkpoint_dir=str(checkpoint_dir),
    )


def _forbid_urlopen(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(url, *args, **kwargs):
        raise AssertionError(f"urlopen must not be called (url={url!r})")

    monkeypatch.setattr("urllib.request.urlopen", _boom)


@pytest.fixture()
def default_env(tmp_path, monkeypatch):
    """Default-checkpoint setup with tmp package/cache dirs and fake payload."""
    package_dir = tmp_path / "pkg_default"
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(Provider, "DEFAULT_CHECKPOINT_DIR", package_dir)
    monkeypatch.setattr(Provider, "EXPECTED_CHECKPOINT_SIZE", len(PAYLOAD))
    monkeypatch.setattr(Provider, "EXPECTED_CHECKPOINT_SHA256", PAYLOAD_SHA256)
    monkeypatch.setattr(
        Provider, "_default_download_cache_dir", staticmethod(lambda: cache_dir)
    )
    provider = _make_provider(package_dir)
    return provider, package_dir, cache_dir


def _cache_target(cache_dir: Path) -> Path:
    return cache_dir / Provider.MODEL_SUBDIR / Provider.MODEL_FILENAME


def _leftover_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.is_file()]


# --------------------------------------------------------------------------
# Pinned-source constants
# --------------------------------------------------------------------------


def test_checkpoint_url_is_pinned_to_immutable_commit():
    assert Provider.CHECKPOINT_UPSTREAM_COMMIT == PINNED_COMMIT
    url = Provider.DEFAULT_CHECKPOINT_URL
    assert PINNED_COMMIT in url
    assert "/main/" not in url
    assert url == (
        "https://raw.githubusercontent.com/elchun/contact_graspnet_pytorch/"
        f"{PINNED_COMMIT}/checkpoints/contact_graspnet/checkpoints/model.pt"
    )
    assert Provider.EXPECTED_CHECKPOINT_SIZE == PINNED_SIZE
    assert Provider.EXPECTED_CHECKPOINT_SHA256 == PINNED_SHA256


# --------------------------------------------------------------------------
# Download paths
# --------------------------------------------------------------------------


def test_download_success_streams_validates_and_replaces(default_env, monkeypatch):
    provider, package_dir, cache_dir = default_env
    calls = []

    def fake_urlopen(url, *args, **kwargs):
        calls.append(url)
        return _FakeResponse(PAYLOAD)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    resolved = provider._ensure_checkpoint_file()

    target = _cache_target(cache_dir)
    assert target.read_bytes() == PAYLOAD
    assert calls == [Provider.DEFAULT_CHECKPOINT_URL]
    # No temporary or stray files remain.
    assert _leftover_files(cache_dir) == [target]
    # The configured checkpoint_dir is never mutated; the resolved path
    # returned for loading points to the cache dir instead.
    assert Path(provider.checkpoint_dir) == package_dir
    assert resolved == cache_dir
    assert not _leftover_files(package_dir)

    # A second call is a cache-validated no-op (no further downloads).
    resolved = provider._ensure_checkpoint_file()
    assert resolved == cache_dir
    assert calls == [Provider.DEFAULT_CHECKPOINT_URL]


def test_download_size_mismatch_cleans_temp(default_env, monkeypatch):
    provider, _, cache_dir = default_env
    # Undersized stream: passes the early oversized-abort check but fails the
    # final exact-size validation. (Oversized streams abort even earlier; see
    # test_oversized_stream_aborts_early_and_cleans_temp.)
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda url, **_: _FakeResponse(PAYLOAD[:-7])
    )

    with pytest.raises(RuntimeError, match="size mismatch"):
        provider._ensure_checkpoint_file()

    assert not _cache_target(cache_dir).exists()
    assert _leftover_files(cache_dir) == []


def test_download_hash_mismatch_cleans_temp(default_env, monkeypatch):
    provider, _, cache_dir = default_env
    tampered = bytearray(PAYLOAD)
    tampered[0] ^= 0xFF
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda url, **_: _FakeResponse(bytes(tampered))
    )

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        provider._ensure_checkpoint_file()

    assert not _cache_target(cache_dir).exists()
    assert _leftover_files(cache_dir) == []


def test_interrupted_download_cleans_temp(default_env, monkeypatch):
    provider, _, cache_dir = default_env
    # Small chunks so the simulated failure happens mid-stream, after the
    # temporary file already holds partial data.
    monkeypatch.setattr(Provider, "CHECKPOINT_CHUNK_SIZE", 4096)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda url, **_: _FakeResponse(PAYLOAD, fail_on_read=3),
    )

    with pytest.raises(RuntimeError, match="Failed to download"):
        provider._ensure_checkpoint_file()

    assert not _cache_target(cache_dir).exists()
    assert _leftover_files(cache_dir) == []


# --------------------------------------------------------------------------
# Cached/default file handling
# --------------------------------------------------------------------------


def test_valid_cached_checkpoint_is_noop_and_preserved(default_env, monkeypatch):
    provider, _, cache_dir = default_env
    target = _cache_target(cache_dir)
    target.parent.mkdir(parents=True)
    target.write_bytes(PAYLOAD)
    _forbid_urlopen(monkeypatch)

    provider._ensure_checkpoint_file()

    assert target.read_bytes() == PAYLOAD


def test_invalid_cached_checkpoint_is_redownloaded(default_env, monkeypatch):
    provider, _, cache_dir = default_env
    target = _cache_target(cache_dir)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"corrupt-partial-download")
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda url, **_: _FakeResponse(PAYLOAD)
    )

    provider._ensure_checkpoint_file()

    assert target.read_bytes() == PAYLOAD
    assert _leftover_files(cache_dir) == [target]


def test_failed_redownload_preserves_prior_cache_file(default_env, monkeypatch):
    """A failed re-download must not destroy the pre-existing target file."""
    provider, _, cache_dir = default_env
    target = _cache_target(cache_dir)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"corrupt-partial-download")
    monkeypatch.setattr(Provider, "CHECKPOINT_CHUNK_SIZE", 4096)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda url, **_: _FakeResponse(PAYLOAD, fail_on_read=2),
    )

    with pytest.raises(RuntimeError, match="Failed to download"):
        provider._ensure_checkpoint_file()

    assert target.read_bytes() == b"corrupt-partial-download"
    assert _leftover_files(cache_dir) == [target]


def test_valid_default_package_checkpoint_is_noop(default_env, monkeypatch):
    provider, package_dir, cache_dir = default_env
    pkg_target = package_dir / Provider.MODEL_SUBDIR / Provider.MODEL_FILENAME
    pkg_target.parent.mkdir(parents=True)
    pkg_target.write_bytes(PAYLOAD)
    _forbid_urlopen(monkeypatch)

    resolved = provider._ensure_checkpoint_file()

    assert pkg_target.read_bytes() == PAYLOAD
    assert resolved == package_dir
    assert Path(provider.checkpoint_dir) == package_dir
    assert not cache_dir.exists()


def test_invalid_default_package_checkpoint_raises(default_env, monkeypatch):
    provider, package_dir, _ = default_env
    pkg_target = package_dir / Provider.MODEL_SUBDIR / Provider.MODEL_FILENAME
    pkg_target.parent.mkdir(parents=True)
    pkg_target.write_bytes(b"corrupt")
    _forbid_urlopen(monkeypatch)

    with pytest.raises(RuntimeError, match="does not match"):
        provider._ensure_checkpoint_file()

    # The package file is never deleted or modified by the provider.
    assert pkg_target.read_bytes() == b"corrupt"


# --------------------------------------------------------------------------
# Custom checkpoint_dir behavior
# --------------------------------------------------------------------------


def test_custom_checkpoint_dir_missing_raises_without_download(
    tmp_path, monkeypatch
):
    provider = _make_provider(tmp_path / "custom")
    _forbid_urlopen(monkeypatch)

    with pytest.raises(FileNotFoundError, match="custom"):
        provider._ensure_checkpoint_file()


def test_custom_checkpoint_dir_existing_file_is_used_as_is(tmp_path, monkeypatch):
    custom = tmp_path / "custom"
    target = custom / Provider.MODEL_SUBDIR / Provider.MODEL_FILENAME
    target.parent.mkdir(parents=True)
    target.write_bytes(b"user-provided-weights")
    provider = _make_provider(custom)
    _forbid_urlopen(monkeypatch)

    resolved = provider._ensure_checkpoint_file()

    # Custom checkpoints are not forced to match the default pinned hash.
    assert target.read_bytes() == b"user-provided-weights"
    assert resolved == custom
    assert Path(provider.checkpoint_dir) == custom


# --------------------------------------------------------------------------
# Network hardening: timeout and oversized-response abort
# --------------------------------------------------------------------------


def test_download_passes_explicit_urlopen_timeout(default_env, monkeypatch):
    provider, _, _ = default_env
    seen = {}

    def fake_urlopen(url, **kwargs):
        seen.update(kwargs)
        return _FakeResponse(PAYLOAD)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    provider._ensure_checkpoint_file()

    assert seen.get("timeout") == Provider.CHECKPOINT_URLOPEN_TIMEOUT
    assert (
        isinstance(Provider.CHECKPOINT_URLOPEN_TIMEOUT, (int, float))
        and Provider.CHECKPOINT_URLOPEN_TIMEOUT > 0
    )


def test_oversized_stream_aborts_early_and_cleans_temp(default_env, monkeypatch):
    provider, _, cache_dir = default_env
    monkeypatch.setattr(Provider, "CHECKPOINT_CHUNK_SIZE", 4096)
    oversized = PAYLOAD + b"x" * 8192  # larger than the patched expected size
    responses = []

    def fake_urlopen(url, **kwargs):
        resp = _FakeResponse(oversized)
        responses.append(resp)
        return resp

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="exceeds the expected"):
        provider._ensure_checkpoint_file()

    # The stream was aborted before being fully consumed.
    assert len(responses) == 1
    assert responses[0]._offset < len(oversized)
    assert not _cache_target(cache_dir).exists()
    assert _leftover_files(cache_dir) == []


# --------------------------------------------------------------------------
# Repeated default-cache validation
# --------------------------------------------------------------------------


def test_corrupted_cache_after_successful_ensure_is_redownloaded(
    default_env, monkeypatch
):
    """Every ensure revalidates the cached default checkpoint.

    Regression: because the configured ``checkpoint_dir`` is immutable and
    default/custom behavior is re-derived from it on every call, a second
    ensure must still treat the provider as default-origin and validate the
    (now corrupted) cache file instead of silently accepting it as a custom
    checkpoint.
    """
    provider, package_dir, cache_dir = default_env
    calls = []

    def fake_urlopen(url, **kwargs):
        calls.append(url)
        return _FakeResponse(PAYLOAD)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    resolved = provider._ensure_checkpoint_file()
    target = _cache_target(cache_dir)
    assert target.read_bytes() == PAYLOAD
    assert len(calls) == 1
    # The configured checkpoint_dir stays the immutable default; the resolved
    # path returned for loading points to the cache.
    assert Path(provider.checkpoint_dir) == package_dir
    assert resolved == cache_dir

    # Corrupt the cache *after* a successful ensure.
    target.write_bytes(b"corrupted-after-first-ensure")

    resolved = provider._ensure_checkpoint_file()
    assert len(calls) == 2, "corrupted cache must trigger a re-download"
    assert target.read_bytes() == PAYLOAD
    assert _leftover_files(cache_dir) == [target]
    assert Path(provider.checkpoint_dir) == package_dir
    assert resolved == cache_dir
