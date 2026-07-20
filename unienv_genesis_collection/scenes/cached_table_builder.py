"""CachedTableBuilder: pre-builds N table variants at reload time and selects
one per environment each reset, without recompiling the scene.

This class subclasses :class:`TableBuilder` and reuses its shared table API
(default_* fields and placement helpers).  Cached variants still follow their
own reload/reset lifecycle.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from unienv_genesis import GenesisWorld
from unienv_genesis.utils.entity_group import EntityGroup
from unienv_genesis_collection.scenes.table_builder import (
    TableBuilder,
    list_all_table_textures,
    _LEG_COLORS,
    DEFAULT_TABLE_SIZE,
    DEFAULT_TABLE_HEIGHT,
    DEFAULT_LEG_RADIUS,
)

# Entities of inactive variants are parked far below the scene so they do not
# interfere with physics or rendering.
_PARK_POS = (0.0, 0.0, -1000.0)
_PARK_QUAT = (1.0, 0.0, 0.0, 0.0)

class CachedTableBuilder(TableBuilder):
    """Pre-builds N table variants during reload(); selects one per reset().

    Unlike :class:`TableBuilder`, which randomises appearance on every
    ``reload()`` (triggering scene recompilation), this class builds all
    variants once and repositions them on each ``reset()`` — no recompilation
    needed between episodes.

    ``size`` and ``height`` are exposed as properties that follow the currently
    active variant selection.  Before the first reset (i.e. before the
    after_reload full reset initialises the per-env variant choices) they
    return the fixed configured values from :class:`TableBuilder`.

    During ``reload()`` (priority 100):
      - Creates ``n_variants`` tables with randomly sampled combinations of
        texture, leg colour, size (within ``size_range``), and height (within
        ``height_range``). All variants are initially placed at
        ``_PARK_POS = (0, 0, -1000)`` so they do not appear in the scene.

    During ``reset()`` (priority 100):
      - In a single environment: selects one variant and moves it to
        ``pos``/``quat``; parks all others.
      - In a batched world: each environment independently selects a variant,
        using batched ``set_pose`` calls.
    """

    has_reward = False
    has_termination_signal = False
    has_truncation_signal = False

    reload_priorities = {100}
    after_reload_priorities = {100}
    reset_priorities = {100}

    @property
    def size(self) -> torch.Tensor:
        if self._sampled_sizes_t is None or self._chosen_indices is None:
            return self._fixed_size_t
        chosen_sizes = self._sampled_sizes_t[self._chosen_indices]
        if self.world.batch_size is None:
            return chosen_sizes[0]
        return chosen_sizes

    @size.setter
    def size(self, value) -> None:
        self._fixed_size_t = torch.as_tensor(
            value, dtype=self.world.dtype, device=self.world.device,
        )

    @property
    def height(self) -> torch.Tensor:
        if self._sampled_heights_t is None or self._chosen_indices is None:
            return self._fixed_height_t
        chosen_heights = self._sampled_heights_t[self._chosen_indices]
        if self.world.batch_size is None:
            return chosen_heights[0]
        return chosen_heights

    @height.setter
    def height(self, value) -> None:
        self._fixed_height_t = torch.as_tensor(
            value, dtype=self.world.dtype, device=self.world.device,
        )

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
        super().__init__(
            world=world,
            name=name,
            size=size,
            height=height,
            leg_radius=leg_radius,
            pos=pos,
            yaw=yaw,
            texture_path=texture_path,
            leg_color=leg_color,
        )
        self.n_variants = n_variants
        self.size_range = size_range
        self.height_range = height_range

        # Persistent RNGs — reseeded when an explicit seed is provided,
        # otherwise their state carries over across reload/reset calls.
        self.np_rng: np.random.Generator = np.random.default_rng()
        self.rng: torch.Generator = torch.Generator(device=world.device)
        self.rng.seed()  # seed from OS entropy

        # Populated during reload().
        self._variants: List[EntityGroup] = []
        self._variant_params: List[dict] = []
        self._sampled_sizes_t: Optional[torch.Tensor] = None   # (n_variants, 3)
        self._sampled_heights_t: Optional[torch.Tensor] = None # (n_variants,)
        # After reset(): LongTensor of shape (1,) (unbatched) or
        # (n_envs,) (batched) on the world's device, giving the chosen
        # variant index for each environment.  ``None`` until the first
        # (full) reset initialises it.
        self._chosen_indices: Optional[torch.Tensor] = None

    def _reseed(self, seed: int) -> None:
        """Reseed both RNGs from *seed*."""
        self.np_rng = np.random.default_rng(seed)
        self.rng.manual_seed(seed)

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
        sampled_sizes: List[Tuple[float, float, float]] = []
        sampled_heights: List[float] = []
        self._chosen_indices = None

        for i in range(self.n_variants):
            variant_name = f"{self._name}_v{i}"

            if self.size_range is not None:
                (w_min, w_max), (d_min, d_max) = self.size_range
                variant_size = (
                    float(self.np_rng.uniform(w_min, w_max)),
                    float(self.np_rng.uniform(d_min, d_max)),
                    self.default_size[2],  # thickness always fixed
                )
            else:
                variant_size = self.default_size

            variant_height = (
                float(self.np_rng.uniform(*self.height_range))
                if self.height_range is not None
                else self.default_height
            )

            variant_texture = (
                self.default_texture_path
                if self.default_texture_path is not None
                else all_textures[self.np_rng.integers(len(all_textures))]
            )

            variant_leg_color = (
                self.default_leg_color
                if self.default_leg_color is not None
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
            self._variant_params.append(
                {
                    "size": variant_size,
                    "height": variant_height,
                    "texture_path": variant_texture,
                    "leg_color": variant_leg_color,
                }
            )
            sampled_sizes.append(variant_size)
            sampled_heights.append(variant_height)

        self._sampled_sizes_t = torch.tensor(
            sampled_sizes, dtype=self.world.dtype, device=self.world.device,
        )
        self._sampled_heights_t = torch.tensor(
            sampled_heights, dtype=self.world.dtype, device=self.world.device,
        )

    def after_reload(self, *, priority: int = 0, mask=None, **kwargs) -> None:
        # Scene is now built. Select initial active variant without reseeding.
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

        n_envs = self.world.batch_size  # None -> unbatched (single environment)

        if n_envs is None:
            # Unbatched: one environment, one active variant (mask N/A).
            # The draw must be requested on the world's device: ``self.rng``
            # is a device-matched generator, and torch raises when the
            # generator device does not match the output device.
            chosen = torch.randint(
                self.n_variants, (1,),
                generator=self.rng, device=self.world.device,
            )
            self._chosen_indices = chosen

            target_pos = torch.tensor(self.pos, dtype=self.world.dtype, device=self.world.device)
            target_quat = torch.tensor(self.quat, dtype=self.world.dtype, device=self.world.device)
            park_pos = torch.tensor(_PARK_POS, dtype=self.world.dtype, device=self.world.device)
            park_quat = torch.tensor(_PARK_QUAT, dtype=self.world.dtype, device=self.world.device)

            for i, group in enumerate(self._variants):
                if i == int(chosen.item()):
                    group.set_pose(target_pos, target_quat)
                else:
                    group.set_pose(park_pos, park_quat)
            return

        # Batched: each masked environment independently picks a variant;
        # unmasked environments keep their current variant and pose.
        # Reset masks are strictly ``(n_envs,)`` torch.bool tensors on the
        # world's device — validated here at the reset boundary.
        if mask is not None:
            assert isinstance(mask, torch.Tensor) and mask.dtype == torch.bool, (
                f"Reset mask must be a torch.bool tensor, got "
                f"{type(mask)} with dtype {getattr(mask, 'dtype', None)}"
            )
            assert mask.shape == (n_envs,), (
                f"Reset mask must have shape ({n_envs},), got {tuple(mask.shape)}"
            )
            assert mask.device == self.world.device, (
                f"Reset mask must live on the world device "
                f"{self.world.device}, got {mask.device}"
            )
            assert self._chosen_indices is not None, (
                "Masked reset before the variant choices were initialised — "
                "run a full reset first (after_reload performs one)."
            )
            n_select = int(mask.sum().item())
            if n_select == 0:
                return
        else:
            n_select = n_envs

        new_choices = torch.randint(
            self.n_variants, (n_select,),
            generator=self.rng, device=self.world.device,
        )

        # Start from the previous selection so unmasked envs keep their
        # variant.  A full reset re-initialises the choices for every env.
        if mask is None:
            chosen_t = new_choices
        else:
            chosen_t = self._chosen_indices.clone()
            chosen_t[mask] = new_choices
        self._chosen_indices = chosen_t

        target_pos = torch.tensor(self.pos, dtype=self.world.dtype, device=chosen_t.device)
        target_quat = torch.tensor(self.quat, dtype=self.world.dtype, device=chosen_t.device)
        park_pos = torch.tensor(_PARK_POS, dtype=self.world.dtype, device=chosen_t.device)
        park_quat = torch.tensor(_PARK_QUAT, dtype=self.world.dtype, device=chosen_t.device)

        for i, group in enumerate(self._variants):
            is_active = (chosen_t == i).unsqueeze(-1)

            positions = torch.where(
                is_active,
                target_pos.expand(n_envs, 3),
                park_pos.expand(n_envs, 3),
            )

            quats = torch.where(
                is_active,
                target_quat.expand(n_envs, 4),
                park_quat.expand(n_envs, 4),
            )

            # Pose tensors cover the full batch; EntityGroup forwards only
            # the masked rows to Genesis (boolean masks are valid Genesis
            # envs_idx), so unmasked envs are untouched.
            group.set_pose(positions, quats, envs_idx=mask)
