"""Ground-truth 3D keypoint trajectory / flow sensor node for Genesis.

This sensor delegates image acquisition to an existing :class:`CameraSensorNode`
that provides segmentation and per-pixel 3D positions. At reset/reload time it:

1. Proposes keypoints from the camera image via a :class:`PointProposer`.
2. Maps each keypoint to a rigid entity using camera segmentation metadata.
3. Stores each keypoint's local offset in the entity frame.

After each environment step, keypoints are updated from entity poses only
(ground-truth tracking), and per-step 3D flow is produced.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Mapping, Optional, Sequence, Set, Tuple, Union

import torch
from genesis.engine.entities.rigid_entity import RigidEntity, RigidLink

from xbarray.transformations.rotation_conversions.pytorch import (
    quaternion_apply,
    quaternion_invert,
)

from unienv_interface.space import DictSpace, BoxSpace
from unienv_interface.backends.pytorch import (
    PyTorchComputeBackend,
    PyTorchArrayType,
    PyTorchDeviceType,
    PyTorchDtypeType,
    PyTorchRNGType,
)

from unienv_genesis.world.genesis_world import GenesisWorld

from .base import SensorNode
from .camera_node import CameraSensorNode

__all__ = [
    "PointProposer",
    "UniformSegmentationPointProposer",
    "GTKeypointFlowSensorNode",
]


class PointProposer(ABC):
    """Base class for selecting keypoint pixels from camera observations."""

    @abstractmethod
    def propose(
        self,
        *,
        segmentation: torch.Tensor,
        position: torch.Tensor,
        valid_mask: torch.Tensor,
        seg_to_entity_idx: Mapping[int, int],
        max_points: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Return ``(N, 2)`` integer tensor of ``[y, x]`` pixels to track."""


class UniformSegmentationPointProposer(PointProposer):
    """Simple point proposer using valid segmented pixels.

    Args:
        max_points_per_entity: Optional cap for points selected from each
            segmentation id.
        randomize: If ``True``, randomly sub-sample candidate pixels.
            If ``False``, keep deterministic raster-scan order.
    """

    def __init__(
        self,
        *,
        max_points_per_entity: Optional[int] = None,
        randomize: bool = False,
    ):
        self.max_points_per_entity = max_points_per_entity
        self.randomize = randomize

    def _choose(
        self,
        pixels: torch.Tensor,
        n: int,
        *,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        if pixels.shape[0] <= n:
            return pixels
        if self.randomize:
            perm = torch.randperm(pixels.shape[0], generator=generator, device=pixels.device)
            return pixels[perm[:n]]
        return pixels[:n]

    def propose(
        self,
        *,
        segmentation: torch.Tensor,
        position: torch.Tensor,
        valid_mask: torch.Tensor,
        seg_to_entity_idx: Mapping[int, int],
        max_points: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        del position  # not needed by this proposer

        if max_points <= 0 or len(seg_to_entity_idx) == 0:
            return torch.empty((0, 2), dtype=torch.long, device=segmentation.device)

        allowed_seg_ids = torch.tensor(
            list(seg_to_entity_idx.keys()),
            dtype=segmentation.dtype,
            device=segmentation.device,
        )
        candidate_mask = valid_mask & torch.isin(segmentation, allowed_seg_ids)
        candidate_pixels = torch.nonzero(candidate_mask, as_tuple=False)  # (K, 2)
        if candidate_pixels.numel() == 0:
            return torch.empty((0, 2), dtype=torch.long, device=segmentation.device)

        if self.max_points_per_entity is None:
            return self._choose(candidate_pixels, max_points, generator=generator)

        seg_values = segmentation[candidate_pixels[:, 0], candidate_pixels[:, 1]]
        entity_values = torch.full_like(seg_values, -1, dtype=torch.int64)
        for seg_id, entity_idx in seg_to_entity_idx.items():
            entity_values[seg_values == int(seg_id)] = int(entity_idx)

        valid_entity = entity_values >= 0
        if not torch.any(valid_entity):
            return torch.empty((0, 2), dtype=torch.long, device=segmentation.device)
        candidate_pixels = candidate_pixels[valid_entity]
        entity_values = entity_values[valid_entity]

        selected: list[torch.Tensor] = []
        remaining = max_points
        for entity_idx in torch.unique(entity_values):
            if remaining <= 0:
                break
            entity_pixels = candidate_pixels[entity_values == entity_idx]
            take = min(self.max_points_per_entity, entity_pixels.shape[0], remaining)
            if take <= 0:
                continue
            selected.append(self._choose(entity_pixels, take, generator=generator))
            remaining -= take

        if len(selected) == 0:
            return torch.empty((0, 2), dtype=torch.long, device=segmentation.device)
        return torch.cat(selected, dim=0)


class GTKeypointFlowSensorNode(SensorNode[None, None, Dict[str, PyTorchArrayType]]):
    """Ground-truth keypoint tracking sensor.

    Notes
    -----
    - This node currently focuses on **RigidEntity objects with a single link**
      (typical rigid props). Multi-link articulated bodies are skipped by default.
    - The delegated camera must provide segmentation (``render_seg=True``) and
      either position (``render_position=True``) or depth.
    """

    reload_priorities = set()
    reset_priorities = set()

    # Must run after delegated camera after_reset/after_reload (-50).
    after_reload_priorities = {-60}
    after_reset_priorities = {-60}

    # Must run after delegated camera post step (0).
    post_environment_step_priorities = {-1}

    def __init__(
        self,
        world: GenesisWorld,
        name: str,
        camera_node: CameraSensorNode,
        *,
        max_points: int = 128,
        point_proposer: Optional[PointProposer] = None,
        tracked_entity_names: Optional[Sequence[str]] = None,
        tracked_entity_name_prefixes: Optional[Sequence[str]] = None,
        allow_articulated: bool = False,
        control_timestep: Optional[float] = None,
        update_timestep: Optional[float] = None,
    ):
        super().__init__(
            world=world,
            name=name,
            control_timestep=control_timestep,
            update_timestep=update_timestep,
        )
        if max_points <= 0:
            raise ValueError(f"max_points must be > 0, got {max_points}")
        if camera_node.world is not world:
            raise ValueError("camera_node must belong to the same world.")
        if not camera_node.render_seg:
            raise ValueError("camera_node must render segmentation (render_seg=True).")
        if not (camera_node.render_position or camera_node.render_depth):
            raise ValueError(
                "camera_node must provide position or depth. "
                "Set render_position=True (recommended) or render_depth=True."
            )

        self.world = world
        self.camera_node = camera_node
        self.max_points = int(max_points)
        self.point_proposer = point_proposer or UniformSegmentationPointProposer()
        self.allow_articulated = allow_articulated
        self.tracked_entity_names: Optional[Set[str]] = (
            set(tracked_entity_names) if tracked_entity_names is not None else None
        )
        self.tracked_entity_name_prefixes: Optional[Tuple[str, ...]] = (
            tuple(tracked_entity_name_prefixes)
            if tracked_entity_name_prefixes is not None
            else None
        )

        self._n_envs = self.world.batch_size if self.world.batch_size is not None else 1
        self._device = self.world.device
        self._entity_by_idx: Dict[int, RigidEntity] = {}
        self._link_by_idx: Dict[int, RigidLink] = {}
        self._rng = torch.Generator(device="cpu")

        self._points_world = torch.zeros((self._n_envs, self.max_points, 3), dtype=torch.float32, device=self._device)
        self._prev_points_world = torch.zeros_like(self._points_world)
        self._flow_world = torch.zeros_like(self._points_world)
        self._local_offsets = torch.zeros_like(self._points_world)
        self._valid_mask = torch.zeros((self._n_envs, self.max_points), dtype=torch.bool, device=self._device)
        self._entity_idx = torch.full((self._n_envs, self.max_points), -1, dtype=torch.int32, device=self._device)
        self._link_idx = torch.full((self._n_envs, self.max_points), -1, dtype=torch.int32, device=self._device)
        self._seg_idxc = torch.full((self._n_envs, self.max_points), -1, dtype=torch.int32, device=self._device)
        self._pixel_yx = torch.full((self._n_envs, self.max_points, 2), -1, dtype=torch.int32, device=self._device)

        self._last_observation: Optional[Dict[str, PyTorchArrayType]] = None
        self.observation_space = self._get_observation_space()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_observation_space(self) -> DictSpace[PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType]:
        H = self.camera_node.option.res[1]
        W = self.camera_node.option.res[0]
        spaces = {
            "points_world": BoxSpace(
                PyTorchComputeBackend,
                low=-float("inf"),
                high=float("inf"),
                shape=(self.max_points, 3),
                dtype=torch.float32,
                device=self._device,
            ),
            "flow_world": BoxSpace(
                PyTorchComputeBackend,
                low=-float("inf"),
                high=float("inf"),
                shape=(self.max_points, 3),
                dtype=torch.float32,
                device=self._device,
            ),
            "valid_mask": BoxSpace(
                PyTorchComputeBackend,
                low=0,
                high=1,
                shape=(self.max_points,),
                dtype=torch.uint8,
                device=self._device,
            ),
            "entity_idx": BoxSpace(
                PyTorchComputeBackend,
                low=-1,
                high=2**31 - 1,
                shape=(self.max_points,),
                dtype=torch.int32,
                device=self._device,
            ),
            "link_idx": BoxSpace(
                PyTorchComputeBackend,
                low=-1,
                high=2**31 - 1,
                shape=(self.max_points,),
                dtype=torch.int32,
                device=self._device,
            ),
            "seg_idxc": BoxSpace(
                PyTorchComputeBackend,
                low=-1,
                high=2**31 - 1,
                shape=(self.max_points,),
                dtype=torch.int32,
                device=self._device,
            ),
            "pixel_yx": BoxSpace(
                PyTorchComputeBackend,
                low=-1,
                high=max(H, W),
                shape=(self.max_points, 2),
                dtype=torch.int32,
                device=self._device,
            ),
        }
        return DictSpace(PyTorchComputeBackend, spaces, device=self._device)

    def _zero_env_slots(self, env_idx: int) -> None:
        self._points_world[env_idx].zero_()
        self._prev_points_world[env_idx].zero_()
        self._flow_world[env_idx].zero_()
        self._local_offsets[env_idx].zero_()
        self._valid_mask[env_idx].zero_()
        self._entity_idx[env_idx].fill_(-1)
        self._link_idx[env_idx].fill_(-1)
        self._seg_idxc[env_idx].fill_(-1)
        self._pixel_yx[env_idx].fill_(-1)

    def _camera_env_indices(self, n_camera_envs: int) -> torch.Tensor:
        """Map camera batch axis to world environment indices."""
        if self.world.batch_size is None:
            return torch.arange(n_camera_envs, dtype=torch.long, device=self._device)

        if n_camera_envs == self._n_envs:
            return torch.arange(self._n_envs, dtype=torch.long, device=self._device)

        sensor = self.camera_node.sensor
        rendered_envs_idx = None
        if hasattr(sensor, "_visualizer") and hasattr(sensor._visualizer, "_context"):
            rendered_envs_idx = getattr(sensor._visualizer._context, "rendered_envs_idx", None)
        if rendered_envs_idx is not None and len(rendered_envs_idx) == n_camera_envs:
            return torch.as_tensor(rendered_envs_idx, dtype=torch.long, device=self._device)

        if n_camera_envs == 1:
            env_idx = self.camera_node.option.env_idx if self.camera_node.option.env_idx is not None else 0
            return torch.tensor([int(env_idx)], dtype=torch.long, device=self._device)

        return torch.arange(min(n_camera_envs, self._n_envs), dtype=torch.long, device=self._device)

    def _get_camera_maps(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = self.camera_node.get_observation()
        if obs is None:
            raise RuntimeError(
                f"Camera node '{self.camera_node.name}' has no observation yet; "
                "ensure this tracking node runs after the camera."
            )
        if "seg" not in obs:
            raise RuntimeError(f"Camera node '{self.camera_node.name}' must provide segmentation ('seg').")

        seg = torch.as_tensor(obs["seg"], device=self._device)
        if "position" in obs:
            position = torch.as_tensor(obs["position"], dtype=torch.float32, device=self._device)
        elif "depth" in obs:
            position = self.camera_node.depth_to_position_map(obs["depth"])
            position = torch.as_tensor(position, dtype=torch.float32, device=self._device)
        else:
            raise RuntimeError(
                f"Camera node '{self.camera_node.name}' must provide position or depth for GT tracking."
            )

        if seg.ndim == 2:
            seg = seg.unsqueeze(0)
        elif seg.ndim != 3:
            raise ValueError(f"Unsupported segmentation shape: {tuple(seg.shape)}")

        if position.ndim == 3 and position.shape[-1] == 3:
            position = position.unsqueeze(0)
        elif position.ndim != 4 or position.shape[-1] != 3:
            raise ValueError(f"Unsupported position map shape: {tuple(position.shape)}")

        if seg.shape[0] != position.shape[0]:
            raise ValueError(
                f"Segmentation and position batch sizes mismatch: {seg.shape[0]} vs {position.shape[0]}"
            )

        env_indices = self._camera_env_indices(seg.shape[0])
        if env_indices.shape[0] != seg.shape[0]:
            # Fallback alignment to the shorter side.
            n = min(env_indices.shape[0], seg.shape[0])
            seg = seg[:n]
            position = position[:n]
            env_indices = env_indices[:n]

        return seg, position, env_indices

    @staticmethod
    def _seg_key_to_entity_and_link(seg_key: Union[int, Tuple[int, ...]]) -> Tuple[int, Optional[int]]:
        if isinstance(seg_key, tuple):
            entity_idx = int(seg_key[0])
            link_idx = int(seg_key[1]) if len(seg_key) >= 2 else None
            return entity_idx, link_idx
        return int(seg_key), None

    def _build_seg_to_entity_idx(self) -> Dict[int, Tuple[int, int]]:
        seg_to_entity_idx: Dict[int, Tuple[int, int]] = {}
        self._entity_by_idx.clear()
        self._link_by_idx.clear()

        sensor = self.camera_node.sensor
        if not hasattr(sensor, "_visualizer"):
            return seg_to_entity_idx

        seg_idx_dict = getattr(sensor._visualizer, "segmentation_idx_dict", None)
        if seg_idx_dict is None:
            return seg_to_entity_idx

        rigid_solver = self.world.scene.rigid_solver
        rigid_entities = rigid_solver.entities
        rigid_links = rigid_solver.links
        n_entities = len(rigid_entities)
        n_links = len(rigid_links)

        def _name_allowed(entity_name: str) -> bool:
            if self.tracked_entity_names is None and self.tracked_entity_name_prefixes is None:
                return True

            exact_ok = (
                self.tracked_entity_names is not None
                and entity_name in self.tracked_entity_names
            )
            prefix_ok = (
                self.tracked_entity_name_prefixes is not None
                and any(entity_name.startswith(p) for p in self.tracked_entity_name_prefixes)
            )
            return exact_ok or prefix_ok

        for seg_idxc, seg_key in seg_idx_dict.items():
            entity_idx, link_idx = self._seg_key_to_entity_and_link(seg_key)
            if entity_idx < 0 or entity_idx >= n_entities:
                continue

            entity = rigid_entities[entity_idx]
            if not isinstance(entity, RigidEntity):
                continue
            if not _name_allowed(entity.name):
                continue

            resolved_link_idx = int(entity.base_link_idx) if link_idx is None else int(link_idx)
            if resolved_link_idx < 0 or resolved_link_idx >= n_links:
                continue

            if not self.allow_articulated:
                if int(getattr(entity, "n_links", 1)) != 1:
                    continue
                if resolved_link_idx != int(entity.base_link_idx):
                    continue

            link = rigid_links[resolved_link_idx]
            if not isinstance(link, RigidLink):
                continue

            seg_to_entity_idx[int(seg_idxc)] = (int(entity_idx), int(resolved_link_idx))
            self._entity_by_idx[int(entity_idx)] = entity
            self._link_by_idx[int(resolved_link_idx)] = link

        return seg_to_entity_idx

    def _populate_slots_from_pixels(
        self,
        *,
        env_idx: int,
        segmentation: torch.Tensor,
        position: torch.Tensor,
        pixels_yx: torch.Tensor,
        seg_to_entity_idx: Mapping[int, Tuple[int, int]],
    ) -> None:
        if pixels_yx.numel() == 0:
            return

        n = min(self.max_points, pixels_yx.shape[0])
        y = pixels_yx[:n, 0].long()
        x = pixels_yx[:n, 1].long()
        seg_ids = segmentation[y, x].to(torch.int64)
        points_world = position[y, x].to(torch.float32)

        entity_indices = torch.full((n,), -1, dtype=torch.int64, device=self._device)
        link_indices = torch.full((n,), -1, dtype=torch.int64, device=self._device)
        for i in range(n):
            assign = seg_to_entity_idx.get(int(seg_ids[i].item()), None)
            if assign is None:
                continue
            entity_indices[i] = int(assign[0])
            link_indices[i] = int(assign[1])

        valid = (entity_indices >= 0) & (link_indices >= 0)
        if not torch.any(valid):
            return

        seg_ids = seg_ids[valid]
        points_world = points_world[valid]
        y = y[valid]
        x = x[valid]
        entity_indices = entity_indices[valid]
        link_indices = link_indices[valid]
        n_valid = entity_indices.shape[0]

        local_offsets = torch.zeros((n_valid, 3), dtype=torch.float32, device=self._device)
        for link_idx in torch.unique(link_indices):
            link_idx_i = int(link_idx.item())
            link = self._link_by_idx.get(link_idx_i, None)
            if link is None:
                continue

            env_query = env_idx if self.world.batch_size is not None else None
            link_pos = torch.as_tensor(link.get_pos(env_query), dtype=torch.float32, device=self._device)
            link_quat = torch.as_tensor(link.get_quat(env_query), dtype=torch.float32, device=self._device)

            mask = link_indices == link_idx
            delta = points_world[mask] - link_pos
            q_inv = quaternion_invert(link_quat.unsqueeze(0)).expand(delta.shape[0], -1)
            local_offsets[mask] = quaternion_apply(q_inv, delta)

        self._valid_mask[env_idx, :n_valid] = True
        self._entity_idx[env_idx, :n_valid] = entity_indices.to(torch.int32)
        self._link_idx[env_idx, :n_valid] = link_indices.to(torch.int32)
        self._seg_idxc[env_idx, :n_valid] = seg_ids.to(torch.int32)
        self._pixel_yx[env_idx, :n_valid, 0] = y.to(torch.int32)
        self._pixel_yx[env_idx, :n_valid, 1] = x.to(torch.int32)
        self._local_offsets[env_idx, :n_valid] = local_offsets
        self._points_world[env_idx, :n_valid] = points_world
        self._prev_points_world[env_idx, :n_valid] = points_world
        self._flow_world[env_idx, :n_valid].zero_()

    def _init_tracks(self, mask: Optional[PyTorchArrayType] = None) -> None:
        seg, position, camera_env_indices = self._get_camera_maps()
        seg_to_entity_and_link = self._build_seg_to_entity_idx()
        seg_to_entity_idx = {
            seg_idxc: entity_idx
            for seg_idxc, (entity_idx, _link_idx) in seg_to_entity_and_link.items()
        }

        if mask is not None and self.world.batch_size is not None:
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self._device)
        else:
            mask_t = None

        if mask_t is None:
            for env_idx in range(self._n_envs):
                self._zero_env_slots(env_idx)
        else:
            reset_envs = torch.where(mask_t)[0]
            for env_idx in reset_envs.tolist():
                self._zero_env_slots(int(env_idx))

        valid_position = torch.isfinite(position).all(dim=-1)

        for batch_i in range(seg.shape[0]):
            env_idx = int(camera_env_indices[batch_i].item())
            if env_idx < 0 or env_idx >= self._n_envs:
                continue
            if mask_t is not None and not bool(mask_t[env_idx].item()):
                continue

            pixels_yx = self.point_proposer.propose(
                segmentation=seg[batch_i],
                position=position[batch_i],
                valid_mask=valid_position[batch_i],
                seg_to_entity_idx=seg_to_entity_idx,
                max_points=self.max_points,
                generator=self._rng,
            )
            self._populate_slots_from_pixels(
                env_idx=env_idx,
                segmentation=seg[batch_i],
                position=position[batch_i],
                pixels_yx=pixels_yx,
                seg_to_entity_idx=seg_to_entity_and_link,
            )

        self._refresh_observation()

    def _update_points_from_entity_pose(self) -> None:
        self._prev_points_world.copy_(self._points_world)

        if not torch.any(self._valid_mask):
            self._flow_world.zero_()
            self._refresh_observation()
            return

        for link_idx in torch.unique(self._link_idx[self._valid_mask]):
            link_idx_i = int(link_idx.item())
            link = self._link_by_idx.get(link_idx_i, None)
            if link is None:
                continue

            mask = self._valid_mask & (self._link_idx == link_idx)
            env_ids, point_ids = torch.nonzero(mask, as_tuple=True)
            if env_ids.numel() == 0:
                continue

            link_pos = torch.as_tensor(link.get_pos(None), dtype=torch.float32, device=self._device)
            link_quat = torch.as_tensor(link.get_quat(None), dtype=torch.float32, device=self._device)

            if link_pos.ndim == 1:
                pos = link_pos.unsqueeze(0).expand(env_ids.numel(), -1)
                quat = link_quat.unsqueeze(0).expand(env_ids.numel(), -1)
            else:
                pos = link_pos[env_ids]
                quat = link_quat[env_ids]

            local = self._local_offsets[env_ids, point_ids]
            self._points_world[env_ids, point_ids] = pos + quaternion_apply(quat, local)

        self._flow_world.zero_()
        self._flow_world[self._valid_mask] = (
            self._points_world[self._valid_mask] - self._prev_points_world[self._valid_mask]
        )
        self._refresh_observation()

    def _refresh_observation(self) -> None:
        obs: Dict[str, PyTorchArrayType] = {
            "points_world": self._points_world,
            "flow_world": self._flow_world,
            "valid_mask": self._valid_mask.to(torch.uint8),
            "entity_idx": self._entity_idx,
            "link_idx": self._link_idx,
            "seg_idxc": self._seg_idxc,
            "pixel_yx": self._pixel_yx,
        }
        if self.world.batch_size is None:
            obs = {k: v[0] for k, v in obs.items()}
        self._last_observation = obs

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def after_reload(self, *, priority: int = 0, mask: Optional[PyTorchArrayType] = None) -> None:
        if priority in self.after_reload_priorities:
            self._init_tracks(mask=mask)

    def after_reset(self, *, priority: int = 0, mask: Optional[PyTorchArrayType] = None) -> None:
        if priority in self.after_reset_priorities:
            self._init_tracks(mask=mask)

    def post_environment_step(self, dt, *, priority: int = 0) -> None:
        del dt
        if priority in self.post_environment_step_priorities:
            self._update_points_from_entity_pose()

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def get_observation(self) -> Dict[str, PyTorchArrayType]:
        return self._last_observation
