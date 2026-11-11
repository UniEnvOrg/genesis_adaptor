"""EnvMapManager — selects the Genesis world's environment map on reload.

Environment maps (HDRIs) provide image-based lighting for rendered scenes.
Genesis bakes the env map into the renderer / camera at **build time** — the
Nyx camera reads ``world.env_map`` when it is created during ``reload()``
(see :meth:`CameraSensorNode._init_sensor_nyx`), and the RayTracer bakes it
into the scene during ``world.reload()`` → ``create_scene()``.

This node updates ``world.env_map`` at the highest reload priority so that
subsequent nodes — in particular Nyx cameras (reload priority 0) — pick up the
newly selected env map when they are (re-)created.

Approved collection env maps can be loaded from the authoritative manifest in
``assets/scenes/env_maps/info.json``::

    from unienv_genesis_collection.scenes.env_map_manager import EnvMapManager

    env_maps = [
        EnvMapManager.from_asset(asset_id)
        for asset_id in EnvMapManager.list_assets()
    ]
    manager = EnvMapManager(world, env_maps=env_maps, selection="random")

Explicit HDR paths are also supported by constructing
:class:`GenesisWorldEnvMapOption` yourself::

    from unienv_genesis.options.light_option import GenesisWorldEnvMapOption

    manager = EnvMapManager(
        world,
        env_maps=[
            GenesisWorldEnvMapOption(hdr_path="/data/hdri/studio_small_09_1k.hdr"),
            GenesisWorldEnvMapOption(hdr_path="/data/hdri/abandoned_factory_1k.hdr"),
        ],
        selection="cycle",
    )

.. note::

   **Nyx renderer** (``backend="nyx"`` cameras): fully supported — the env map
   is read at camera-creation time during ``reload()``, which runs *after*
   this node's ``reload()`` (priority 150 > 0).

   **RayTracer**: the env map is baked into the scene during
   ``world.reload()`` → ``create_scene()``, which runs *before* any node
   ``reload()``. The change therefore takes effect on the **next** reload.
   For true per-reload RayTracer selection the scene would need to be
   re-created after this node runs, which is not done here to avoid
   invalidating entities added by higher-priority nodes.

   **BatchRenderer**: does not support environment maps at all.

   In batched worlds a **single** env map is selected per reload and shared
   across all parallel environments (the Nyx plugin's per-env env-map index
   is hardcoded to the parallel-env id, so independent per-env selection is
   not possible without patching the plugin).
"""

from __future__ import annotations

import functools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import torch
from unienv_interface.world.node import WorldNode

from unienv_genesis.options.light_option import GenesisWorldEnvMapOption
from unienv_genesis.world.genesis_world import GenesisWorld
from unienv_genesis_collection import asset_path

__all__ = ["EnvMapManager"]

LOGGER = logging.getLogger(__name__)

# Priority 150 sits in the environment-setup range [100, 200) and runs before
# the table (100), robot (50), clutter (25), and cameras (0). Higher priority
# runs first in the descending-priority reload order.
_ENV_MAP_RELOAD_PRIORITY = 150

_ENV_MAP_ASSET_SUBDIR = Path("scenes") / "env_maps"
_ENV_MAP_MANIFEST_NAME = "info.json"
_APPROVED_STATUS = "approved"
_SELECTION_MODES = ("random", "cycle", "first")


@dataclass(frozen=True)
class _EnvMapAssetRecord:
    asset_id: str
    hdr_relative_path: str
    hdr_path: Path
    rotation_deg: float
    status: str


def _env_map_asset_dir() -> Path:
    return (asset_path.ASSETS_DIR / _ENV_MAP_ASSET_SUBDIR).resolve()


def _env_map_local_path(relative_path: str | Path) -> Path:
    return (_env_map_asset_dir() / relative_path).resolve()


def _resolve_env_map_manifest_path() -> Path:
    return asset_path.get_asset_path(_ENV_MAP_ASSET_SUBDIR, _ENV_MAP_MANIFEST_NAME)


def _load_env_map_manifest() -> dict[str, _EnvMapAssetRecord]:
    """Load env-map manifest entries keyed by asset id."""
    manifest_path = _resolve_env_map_manifest_path()

    with manifest_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, dict):
        raise ValueError(
            f"Env map manifest has invalid or missing 'entries': {manifest_path}"
        )

    entries: dict[str, _EnvMapAssetRecord] = {}
    for asset_id, raw_entry in raw_entries.items():
        if not isinstance(raw_entry, dict):
            raise ValueError(
                f"Env map manifest entry '{asset_id}' must be a JSON object."
            )

        hdr_rel_path = raw_entry.get("hdr_path")
        if not isinstance(hdr_rel_path, str) or not hdr_rel_path:
            raise ValueError(
                f"Env map manifest entry '{asset_id}' is missing hdr_path."
            )

        rotation_deg = float(raw_entry.get("rotation_deg", 0.0))
        status = str(raw_entry.get("status", ""))
        entries[asset_id] = _EnvMapAssetRecord(
            asset_id=asset_id,
            hdr_relative_path=hdr_rel_path,
            hdr_path=_env_map_local_path(hdr_rel_path),
            rotation_deg=rotation_deg,
            status=status,
        )

    return entries


def _infer_env_map_id(hdr_path: str | Path | None) -> Optional[str]:
    """Infer a stable id for an env map from the manifest or filename."""
    if hdr_path is None:
        return None
    return _infer_env_map_id_cached(str(hdr_path))


@functools.lru_cache(maxsize=256)
def _infer_env_map_id_cached(hdr_path: str) -> str:
    # Cached: ``get_info`` asks every step, and reading the manifest (and resolving each entry's path) took about
    # 30 ms; the manifest is a static asset.
    resolved_hdr_path = Path(hdr_path).resolve()
    try:
        for record in _load_env_map_manifest().values():
            if record.hdr_path == resolved_hdr_path:
                return record.asset_id
    except (FileNotFoundError, ValueError):
        pass

    return resolved_hdr_path.stem


class EnvMapManager(WorldNode):
    """Select an environment map for the world on each reload.

    Args:
        world: :class:`GenesisWorld` instance.
        env_maps: Sequence of :class:`GenesisWorldEnvMapOption` objects to pick
            from. Every option must provide ``hdr_path`` so Nyx can load it.
        selection: Selection policy. ``"random"`` samples with the internal
            RNG, ``"cycle"`` iterates through the pool in order, and
            ``"first"`` always picks index 0.
        name: Node name.
        reload_priority: Override the reload priority (default 150).
        seed: Optional fixed seed for the internal RNG.
    """

    has_reward = False
    has_termination_signal = False
    has_truncation_signal = False

    reload_priorities = {_ENV_MAP_RELOAD_PRIORITY}

    @staticmethod
    def list_assets() -> list[str]:
        """Return approved env-map asset ids from the authoritative manifest."""
        return [
            asset_id
            for asset_id, record in _load_env_map_manifest().items()
            if record.status == _APPROVED_STATUS
        ]

    @staticmethod
    def from_asset(asset_id: str) -> GenesisWorldEnvMapOption:
        """Build an env-map option from an approved manifest asset id."""
        records = _load_env_map_manifest()
        record = records.get(asset_id)
        if record is None:
            raise ValueError(f"Unknown env map asset id: {asset_id}")
        if record.status != _APPROVED_STATUS:
            raise ValueError(
                f"Env map asset '{asset_id}' is not approved: status={record.status!r}"
            )
        try:
            hdr_path = asset_path.get_asset_path(
                _ENV_MAP_ASSET_SUBDIR,
                record.hdr_relative_path,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Env map HDR file not found for asset '{asset_id}': {record.hdr_path}"
            ) from exc

        return GenesisWorldEnvMapOption(
            hdr_path=str(hdr_path),
            hdr_rotation=record.rotation_deg,
        )

    def __init__(
        self,
        world: GenesisWorld,
        env_maps: Sequence[GenesisWorldEnvMapOption],
        selection: Literal["random", "cycle", "first"] = "random",
        name: str = "env_map",
        reload_priority: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.world = world
        self._name = name

        if selection not in _SELECTION_MODES:
            raise ValueError(
                f"Invalid env-map selection mode {selection!r}. "
                f"Expected one of {_SELECTION_MODES}."
            )
        self.selection = selection

        if reload_priority is not None:
            self.reload_priorities = {reload_priority}

        pool = list(env_maps)
        if not pool:
            raise ValueError("EnvMapManager requires at least one env map.")

        for i, opt in enumerate(pool):
            if opt.hdr_path is None:
                raise ValueError(
                    f"Env map #{i} has hdr_path=None. EnvMapManager requires "
                    "HDR paths for Nyx-based lighting."
                )

        self._pool: list[GenesisWorldEnvMapOption] = pool

        self.rng: torch.Generator = torch.Generator(device=world.device)
        self.rng.seed()
        if seed is not None:
            self.rng.manual_seed(seed)

        self._current_index: Optional[int] = None
        self._current_env_map: Optional[GenesisWorldEnvMapOption] = None
        self._next_cycle_index = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def pool_size(self) -> int:
        """Number of env maps in the pool."""
        return len(self._pool)

    @property
    def current_index(self) -> Optional[int]:
        """Index of the currently selected env map, or ``None`` before reload."""
        return self._current_index

    @property
    def current_env_map(self) -> Optional[GenesisWorldEnvMapOption]:
        """The currently selected env map option, or ``None`` before reload."""
        return self._current_env_map

    @property
    def current_env_map_id(self) -> Optional[str]:
        """Identifier for the current env map, from the manifest or filename."""
        if self._current_env_map is None or self._current_env_map.hdr_path is None:
            return None
        return _infer_env_map_id(self._current_env_map.hdr_path)

    def _reseed(self, seed: int) -> None:
        self.rng.manual_seed(seed)

    def _select_index(self) -> int:
        if len(self._pool) == 1:
            return 0

        if self.selection == "random":
            return int(
                torch.randint(
                    len(self._pool),
                    (),
                    generator=self.rng,
                    device=self.world.device,
                ).item()
            )

        if self.selection == "cycle":
            idx = self._next_cycle_index % len(self._pool)
            self._next_cycle_index = (idx + 1) % len(self._pool)
            return idx

        return 0

    def reload(self, *, priority=0, seed=None, mask=None, **kwargs):
        if priority not in self.reload_priorities:
            return

        if seed is not None:
            self._reseed(seed)

        idx = self._select_index()
        self._current_index = idx
        self._current_env_map = self._pool[idx]

        # Mutate the world's env map so that Nyx cameras (re-)created later in
        # the reload sequence pick up the newly selected env map.
        self.world.env_map = self._current_env_map
        LOGGER.debug(
            "EnvMapManager '%s' selected env map %d/%d via %s: %s",
            self._name,
            idx,
            len(self._pool),
            self.selection,
            self._current_env_map.hdr_path,
        )

    def get_observation(self):
        return None

    def get_info(self) -> dict[str, Any]:
        """Return metadata for the current env-map selection."""
        return {
            "env_map_selection": self.selection,
            "env_map_index": self._current_index,
            "env_map_id": self.current_env_map_id,
            "env_map_path": (
                str(self._current_env_map.hdr_path)
                if self._current_env_map is not None
                and self._current_env_map.hdr_path is not None
                else None
            ),
            "env_map_rotation_deg": (
                self._current_env_map.hdr_rotation
                if self._current_env_map is not None
                else None
            ),
            "env_map_pool_size": len(self._pool),
        }
