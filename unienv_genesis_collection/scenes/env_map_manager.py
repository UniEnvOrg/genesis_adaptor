"""EnvMapManager — randomizes the Genesis world's environment map on reload.

Environment maps (HDRIs) provide image-based lighting for rendered scenes.
Genesis bakes the env map into the renderer / camera at **build time** — the
Nyx camera reads ``world.env_map`` when it is created during ``reload()``
(see :meth:`CameraSensorNode._init_sensor_nyx`), and the RayTracer bakes it
into the scene during ``world.reload()`` → ``create_scene()``.

This node randomizes ``world.env_map`` at the highest reload priority so that
subsequent nodes — in particular Nyx cameras (reload priority 0) — pick up the
newly selected env map when they are (re-)created.

.. note::

   **Nyx renderer** (``backend="nyx"`` cameras): fully supported — the env map
   is read at camera-creation time during ``reload()``, which runs *after*
   this node's ``reload()`` (priority 150 > 0).

   **RayTracer**: the env map is baked into the scene during
   ``world.reload()`` → ``create_scene()``, which runs *before* any node
   ``reload()``.  The change therefore takes effect on the **next** reload.
   For true per-reload RayTracer randomization the scene would need to be
   re-created after this node runs, which is not done here to avoid
   invalidating entities added by higher-priority nodes.

   **BatchRenderer**: does not support environment maps at all.

   In batched worlds a **single** env map is selected per reload and shared
   across all parallel environments (the Nyx plugin's per-env env-map index
   is hardcoded to the parallel-env id, so independent per-env selection is
   not possible without patching the plugin).

Auto-discovery
--------------

When neither ``env_maps`` nor ``hdr_paths`` is provided, the manager
auto-discovers all ``.hdr`` files in the collection's env-map asset folder
(``assets/scenes/env_maps/``).  Download HDRIs there with::

    python scripts/prepare_env_maps.py --count 10

Example::

    from unienv_genesis_collection.scenes.env_map_manager import EnvMapManager

    # Auto-discover all .hdr files in assets/scenes/env_maps/
    manager = EnvMapManager(world)

    # Or pass explicit paths
    manager = EnvMapManager(
        world,
        hdr_paths=[
            "/data/hdri/studio_small_09_1k.hdr",
            "/data/hdri/abandoned_factory_1k.hdr",
        ],
    )
    env = WorldEnv(world, [plane, table, robot, camera, manager], ...)
    # Each env.reset(reload=True) picks a random env map.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import torch
from unienv_interface.world.node import WorldNode

from unienv_genesis.options.light_option import GenesisWorldEnvMapOption
from unienv_genesis.world.genesis_world import GenesisWorld
from unienv_genesis_collection.asset_path import ASSETS_DIR

__all__ = ["EnvMapManager"]

LOGGER = logging.getLogger(__name__)

# Priority 150 sits in the environment-setup range [100, 200) and runs before
# the table (100), robot (50), clutter (25), and cameras (0).  Higher priority
# runs first in the descending-priority reload order.
_ENV_MAP_RELOAD_PRIORITY = 150

# Default search directory for auto-discovered HDRIs.
_ENV_MAP_ASSET_DIR = ASSETS_DIR / "scenes" / "env_maps"

_DOWNLOAD_HINT = (
    "Download HDRIs with: "
    "python scripts/prepare_env_maps.py --count 10"
)


def _discover_hdr_files() -> List[str]:
    """Return sorted absolute paths of all ``.hdr`` files in the asset folder.

    Raises:
        ValueError: If the asset folder does not exist or contains no
            ``.hdr`` files.
    """
    env_map_dir = _ENV_MAP_ASSET_DIR
    if not env_map_dir.is_dir():
        raise ValueError(
            f"Env map asset directory not found: {env_map_dir}. "
            f"{_DOWNLOAD_HINT}"
        )
    paths = sorted(str(p) for p in env_map_dir.glob("*.hdr"))
    if not paths:
        raise ValueError(
            f"No .hdr files found in {env_map_dir}. {_DOWNLOAD_HINT}"
        )
    return paths


class EnvMapManager(WorldNode):
    """Randomly select an environment map for the world on each reload.

    Holds a pool of :class:`GenesisWorldEnvMapOption` (or HDR file paths which
    are wrapped automatically) and assigns a randomly-chosen one to
    ``world.env_map`` during ``reload()``.

    Args:
        world: :class:`GenesisWorld` instance.
        env_maps: Optional sequence of :class:`GenesisWorldEnvMapOption`.
            When ``None`` (default), ``hdr_paths`` is used.
        hdr_paths: Optional sequence of HDR file paths (str / Path).  Each is
            wrapped in a :class:`GenesisWorldEnvMapOption` with
            ``hdr_path`` set.  Combined with *env_maps* if both are given.
            When both *env_maps* and *hdr_paths* are ``None`` (the default),
            the manager auto-discovers all ``.hdr`` files in the collection's
            env-map asset folder (``assets/scenes/env_maps/``).
        shuffle: When ``True`` (default) the env map is re-randomised on every
            reload.  When ``False`` the first env map in the pool is used
            deterministically (useful for debugging).
        name: Node name.
        reload_priority: Override the reload priority (default 150).
        seed: Optional fixed seed for the internal RNG (mainly for tests).
    """

    has_reward = False
    has_termination_signal = False
    has_truncation_signal = False

    reload_priorities = {_ENV_MAP_RELOAD_PRIORITY}

    def __init__(
        self,
        world: GenesisWorld,
        env_maps: Optional[Sequence[GenesisWorldEnvMapOption]] = None,
        hdr_paths: Optional[Sequence[Union[str, os.PathLike]]] = None,
        shuffle: bool = True,
        name: str = "env_map",
        reload_priority: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.world = world
        self._name = name
        self.shuffle = shuffle

        if reload_priority is not None:
            self.reload_priorities = {reload_priority}

        # Build the pool from both sources; auto-discover when neither is given.
        pool: List[GenesisWorldEnvMapOption] = []
        if env_maps is not None:
            pool.extend(env_maps)
        if hdr_paths is None and not pool:
            # Neither source provided — auto-discover from the asset folder.
            hdr_paths = _discover_hdr_files()
        if hdr_paths is not None:
            for p in hdr_paths:
                pool.append(GenesisWorldEnvMapOption(hdr_path=str(p)))

        if not pool:
            raise ValueError(
                "EnvMapManager requires at least one env map, but none were "
                f"found. {_DOWNLOAD_HINT}"
            )

        # Validate that every option can actually be used for Nyx lighting.
        for i, opt in enumerate(pool):
            if opt.hdr_path is None:
                raise ValueError(
                    f"Env map #{i} has hdr_path=None. EnvMapManager "
                    f"requires HDR paths for Nyx-based lighting."
                )

        self._pool: List[GenesisWorldEnvMapOption] = pool

        # RNG state.
        self.rng: torch.Generator = torch.Generator(device=world.device)
        self.rng.seed()
        if seed is not None:
            self.rng.manual_seed(seed)

        # Current selection (populated after the first reload).
        self._current_index: Optional[int] = None
        self._current_env_map: Optional[GenesisWorldEnvMapOption] = None

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

    def _reseed(self, seed: int) -> None:
        self.rng.manual_seed(seed)

    def reload(self, *, priority=0, seed=None, mask=None, **kwargs):
        if priority not in self.reload_priorities:
            return

        if seed is not None:
            self._reseed(seed)

        if self.shuffle and len(self._pool) > 1:
            idx = int(
                torch.randint(
                    len(self._pool), (), generator=self.rng, device=self.world.device,
                ).item()
            )
        else:
            idx = 0

        self._current_index = idx
        self._current_env_map = self._pool[idx]
        # Mutate the world's env map so that Nyx cameras (re-)created later
        # in the reload sequence pick up the new selection.
        self.world.env_map = self._current_env_map
        LOGGER.debug(
            "EnvMapManager '%s' selected env map %d/%d: %s",
            self._name, idx, len(self._pool), self._current_env_map.hdr_path,
        )

    def get_observation(self):
        return None

    def get_info(self) -> dict:
        """Return the current selection index and HDR path."""
        return {
            "env_map_index": self._current_index,
            "env_map_path": (
                self._current_env_map.hdr_path
                if self._current_env_map is not None
                else None
            ),
            "env_map_pool_size": len(self._pool),
        }
