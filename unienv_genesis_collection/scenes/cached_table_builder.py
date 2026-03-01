"""CachedTableBuilder: pre-builds N table variants at reload time and selects
one per environment each reset, without recompiling the scene."""

import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from unienv_genesis import GenesisWorld
from unienv_genesis.utils.entity_group import EntityGroup
from unienv_interface.world.node import WorldNode
from unienv_genesis_collection.scenes.table_builder import (
    TableBuilder,
    list_all_table_textures,
    yaw_to_quat,
    _LEG_COLORS,
    DEFAULT_TABLE_SIZE,
    DEFAULT_TABLE_HEIGHT,
    DEFAULT_LEG_RADIUS,
)
from xbarray.transformations.rotation_conversions.pytorch import quaternion_multiply

# Entities of inactive variants are parked far below the scene so they do not
# interfere with physics or rendering.
_PARK_POS = (0.0, 0.0, -1000.0)
_PARK_QUAT = (1.0, 0.0, 0.0, 0.0)


class CachedTableBuilder(WorldNode):
    """Pre-builds N table variants during reload(); selects one per reset().

    Unlike :class:`TableBuilder`, which randomises appearance on every
    ``reload()`` (triggering scene recompilation), this class builds all
    variants once and repositions them on each ``reset()`` — no recompilation
    needed between episodes.

    During ``reload()`` (priority 100):
      - Creates ``n_variants`` tables with randomly sampled combinations of
        texture, leg colour, size (within ``size_range``), and height (within
        ``height_range``).  All variants are initially placed at
        ``_PARK_POS = (0, 0, -1000)`` so they do not appear in the scene.

    During ``reset()`` (priority 100, runs before robots at priority 50):
      - In a single environment: selects one variant and moves it to
        ``pos``/``quat``; parks all others.
      - In a batched world (``world.batch_size = n_envs``): each environment
        independently selects a variant.  Uses batched ``set_pose`` calls so
        every environment can show a different table simultaneously.

    Args:
        world: :class:`GenesisWorld` instance.
        name: Base entity name; variants are named ``{name}_v0``, ``{name}_v1``, …
        n_variants: Number of distinct table variants to pre-build. Default 10.
        pos: Target world position ``(x, y, z)`` of the active table's anchor
            (surface centre, matching :class:`TableBuilder`'s convention).
        yaw: Rotation around the Z axis in radians. Default is 0 (no rotation).
        size_range: Optional ``((min_w, max_w), (min_d, max_d))`` in metres.
            Tabletop thickness is always fixed at ``size[2]``.
            If ``None``, all variants use the fixed ``size`` parameter.
        size: Fixed tabletop ``(width, depth, thickness)``. Used when
            ``size_range`` is ``None``. Defaults to ``DEFAULT_TABLE_SIZE``.
        height_range: Optional ``(min_h, max_h)`` in metres.
            If ``None``, all variants use the fixed ``height`` parameter.
        height: Fixed height (surface to floor). Used when ``height_range``
            is ``None``. Defaults to ``DEFAULT_TABLE_HEIGHT``.
        leg_radius: Leg radius in metres, shared across all variants.
        texture_path: If set, all variants share this texture (no
            per-variant texture randomisation).
        leg_color: If set (RGBA tuple), all variants share this leg colour.

    Example::

        world = GenesisWorld(batch_size=16)
        table = CachedTableBuilder(
            world,
            name="table",
            n_variants=12,
            pos=(0.0, 0.0, 0.65),
            size_range=((0.8, 1.2), (0.5, 0.7)),
            height_range=(0.55, 0.75),
        )
        # First reset triggers reload -> builds 12 variants, then selects one
        # per environment.  Subsequent resets just reposition — no recompile.
    """

    has_reward = False
    has_termination_signal = False
    has_truncation_signal = False

    reload_priorities = {100}
    after_reload_priorities = {100}
    reset_priorities = {100}

    def __init__(
        self,
        world: GenesisWorld,
        name: str = "table",
        n_variants: int = 10,
        pos: tuple = (0.0, 0.0, 0.0),
        yaw: float = 0.0,
        size_range: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        size: tuple = DEFAULT_TABLE_SIZE,
        height_range: Optional[Tuple[float, float]] = None,
        height: float = DEFAULT_TABLE_HEIGHT,
        leg_radius: float = DEFAULT_LEG_RADIUS,
        texture_path: Optional[Path] = None,
        leg_color: Optional[tuple] = None,
    ):
        self.world = world
        self._name = name
        self.n_variants = n_variants
        self.pos = pos
        self.yaw = yaw
        self.quat = yaw_to_quat(yaw)
        self._cos_yaw = math.cos(yaw)
        self._sin_yaw = math.sin(yaw)
        self._pos_t = torch.tensor(pos, dtype=torch.float32)
        self._quat_t = torch.tensor(self.quat, dtype=torch.float32)
        self.size_range = size_range
        self.size = size
        self.height_range = height_range
        self.height = height
        self.leg_radius = leg_radius
        self.texture_path = texture_path
        self.leg_color = leg_color

        # Persistent RNGs — reseeded when an explicit seed is provided,
        # otherwise their state carries over across reload/reset calls.
        self.np_rng: np.random.Generator = np.random.default_rng()
        self.rng: torch.Generator = torch.Generator(device=world.device)
        self.rng.seed()  # seed from OS entropy

        # Populated during reload().
        self._variants: List[EntityGroup] = []
        # Per-variant metadata (size, height, texture, leg_color) for introspection.
        self._variant_params: List[dict] = []
        # After reset(): list of length 1 (unbatched) or n_envs (batched),
        # giving the chosen variant index for each environment.
        self._chosen_indices: Optional[List[int]] = None

    @property
    def name(self) -> str:
        return self._name

    def _reseed(self, seed: int) -> None:
        """Reseed both RNGs from *seed*."""
        self.np_rng = np.random.default_rng(seed)
        self.rng.manual_seed(seed)

    @property
    def active_group(self) -> Optional[EntityGroup]:
        """EntityGroup of the active variant for env 0 (or the only environment).

        Returns ``None`` before the first ``reset()``.
        """
        if self._chosen_indices is None:
            return None
        return self._variants[self._chosen_indices[0]]

    def placement_world_pose(
        self,
        local_pos: torch.Tensor,
        local_quat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert a table-local position/quaternion to world frame.

        The local frame origin is at the table surface center (``self.pos``),
        with X/Y axes rotated by ``self.yaw`` around Z.

        Args:
            local_pos: ``(..., 3)`` or ``(..., 2)`` position in table-local
                frame.  If 2-D, Z defaults to ``self.pos[2]``.
            local_quat: ``(..., 4)`` quaternion ``(w, x, y, z)`` in the
                table-local frame.  If ``None``, uses the table's own
                orientation.

        Returns:
            world_pos: ``(..., 3)`` world-frame position.
            world_quat: ``(..., 4)`` world-frame quaternion ``(w, x, y, z)``.
        """
        c, s = self._cos_yaw, self._sin_yaw

        if local_pos.shape[-1] == 2:
            lx = local_pos[..., 0]
            ly = local_pos[..., 1]
            wx = self._pos_t[0] + lx * c - ly * s
            wy = self._pos_t[1] + lx * s + ly * c
            wz = torch.full_like(wx, self._pos_t[2].item())
            world_pos = torch.stack([wx, wy, wz], dim=-1)
        else:
            lx = local_pos[..., 0]
            ly = local_pos[..., 1]
            lz = local_pos[..., 2]
            wx = self._pos_t[0] + lx * c - ly * s
            wy = self._pos_t[1] + lx * s + ly * c
            wz = self._pos_t[2] + lz
            world_pos = torch.stack([wx, wy, wz], dim=-1)

        if local_quat is None:
            world_quat = self._quat_t.expand(world_pos.shape[:-1] + (4,))
        else:
            world_quat = quaternion_multiply(self._quat_t, local_quat)

        return world_pos, world_quat

    def reload(
        self,
        *,
        priority: int = 0,
        seed: Optional[int] = None,
        mask=None,
        **kwargs,
    ) -> None:
        if priority != 100:
            return

        if seed is not None:
            self._reseed(seed)

        all_textures = list_all_table_textures()
        # In a batched world, Genesis requires batch_fixed_verts=True on fixed morphs
        # to allow per-environment position overrides via entity.set_pos().
        batch_fixed_verts = self.world.batch_size is not None

        self._variants = []
        self._variant_params = []
        self._chosen_indices = None

        for i in range(self.n_variants):
            variant_name = f"{self._name}_v{i}"

            if self.size_range is not None:
                (w_min, w_max), (d_min, d_max) = self.size_range
                variant_size = (
                    float(self.np_rng.uniform(w_min, w_max)),
                    float(self.np_rng.uniform(d_min, d_max)),
                    self.size[2],  # thickness always fixed
                )
            else:
                variant_size = self.size

            variant_height = (
                float(self.np_rng.uniform(*self.height_range))
                if self.height_range is not None
                else self.height
            )

            variant_texture = (
                self.texture_path
                if self.texture_path is not None
                else all_textures[self.np_rng.integers(len(all_textures))]
            )

            variant_leg_color = (
                self.leg_color
                if self.leg_color is not None
                else _LEG_COLORS[self.np_rng.integers(len(_LEG_COLORS))]
            )

            group = TableBuilder.create_table(
                world=self.world,
                name=variant_name,
                size=variant_size,
                height=variant_height,
                leg_radius=self.leg_radius,
                pos=_PARK_POS,
                yaw=0.0,
                texture_path=variant_texture,
                leg_color=variant_leg_color,
                batch_fixed_verts=batch_fixed_verts,
            )

            self._variants.append(group)
            self._variant_params.append({
                "size": variant_size,
                "height": variant_height,
                "texture": variant_texture,
                "leg_color": variant_leg_color,
            })

    def after_reload(self, *, priority: int = 0, mask=None, **kwargs) -> None:
        # Scene is now built (world.after_reload() has run). Select the initial
        # active variant using the current RNG state — seed=None so we continue
        # from where reload() left off rather than reseeding.
        if priority != 100:
            return
        self.reset(priority=priority, seed=None, mask=mask, **kwargs)

    def reset(
        self,
        *,
        priority: int = 0,
        seed: Optional[int] = None,
        mask=None,
        **kwargs,
    ) -> None:
        if priority != 100:
            return

        if not self._variants:
            return  # reload() not yet called; defensive guard

        if seed is not None:
            self._reseed(seed)

        n_envs = self.world.batch_size  # None → unbatched (single environment)

        if n_envs is None:
            # Unbatched: one environment, one active variant.
            chosen = int(torch.randint(self.n_variants, (1,), generator=self.rng).item())
            self._chosen_indices = [chosen]

            for i, group in enumerate(self._variants):
                if i == chosen:
                    group.set_pose(self.pos, self.quat)
                else:
                    group.set_pose(_PARK_POS, _PARK_QUAT)
        else:
            # Batched: each environment independently picks a variant.
            # For variant i, build a (n_envs, 3)/(n_envs, 4) pose tensor:
            #   - target pos/quat for environments that chose variant i
            #   - park pos/quat for all others
            chosen_t = torch.randint(self.n_variants, (n_envs,), generator=self.rng)  # (n_envs,)
            self._chosen_indices = chosen_t.tolist()

            target_pos = torch.tensor(self.pos, dtype=torch.float32)    # (3,)
            target_quat = torch.tensor(self.quat, dtype=torch.float32)  # (4,)
            park_pos = torch.tensor(_PARK_POS, dtype=torch.float32)     # (3,)
            park_quat = torch.tensor(_PARK_QUAT, dtype=torch.float32)   # (4,)

            for i, group in enumerate(self._variants):
                is_active = (chosen_t == i).unsqueeze(-1)  # (n_envs, 1) bool

                positions = torch.where(
                    is_active,
                    target_pos.expand(n_envs, 3),
                    park_pos.expand(n_envs, 3),
                )  # (n_envs, 3)

                quats = torch.where(
                    is_active,
                    target_quat.expand(n_envs, 4),
                    park_quat.expand(n_envs, 4),
                )  # (n_envs, 4)

                group.set_pose(positions, quats)
