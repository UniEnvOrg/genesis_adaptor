"""Offline tests for :mod:`unienv_genesis_collection.objects.articulated_object`.

Fully offline: no asset downloads, no Genesis scene builds.  Synthetic URDF
and MJCF assets (tiny box meshes written to ``tmp_path``) exercise the FK
helpers; a lightweight fake entity exposing the same call signatures as a
built ``RigidEntity`` exercises the runtime name resolvers, the
open-fraction helper, per-link poses and part-aware grasp candidates.

Covers:
  - resolve_joint_dofs / resolve_link_index (success + error listing names)
  - urdf_link_meshes_at_config: FK placement, mesh scale, union bbox,
    link_local mode (raw visual geometry per link frame)
  - mjcf_link_meshes_at_config: hinge / slide FK, compiler meshdir, euler,
    link_local mode
  - get_local_bounding_box / get_link_local_bounding_box caching + union
  - get/set of DoFs by joint name (unbatched & batched + envs_idx masks)
  - get_open_fraction / set_open_fraction math, clamping, ±inf limits
  - open-fraction masks: limits read unmasked (batch_dofs_info footgun),
    per-env limits under batch_dofs_info=True
  - get_link_pose batching
  - list_part_grasping_candidates world transform, aliases, error paths,
    masked env subsets and degenerate flat links (empty candidates)
  - part grasps on a link OFFSET/ROTATED from the base (link-frame
    enumeration, no double-applied base_T_link) and slide-joint tracking
  - real-Genesis check (CPU + CUDA): candidates sit on the offset link's
    actual geometry and track the moved joint
  - set_pos_3d defaulting to a non-randomised quaternion
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from unienv_genesis_collection.objects.articulated_object import (
    ArticulatedObject,
    ArticulationError,
    mjcf_link_meshes_at_config,
    resolve_joint_dofs,
    resolve_link_index,
    urdf_link_meshes_at_config,
)
from unienv_genesis_collection.objects.base_object import GraspCandidates

# ------------------------------------------------------------------
# Synthetic assets
# ------------------------------------------------------------------

# Two-link articulated URDF: a base, a prismatic drawer (translation along
# +x) and a revolute door (rotation about +z through the base origin).  Each
# moving link's visual mesh is offset from its link origin, so the FK
# transforms are observable.
_DRAWER_DOOR_URDF = """\
<robot name="unit_drawer_door">
    <link name="base">
        <visual>
            <geometry><mesh filename="base.obj" scale="1 1 1" /></geometry>
        </visual>
    </link>
    <link name="drawer">
        <visual>
            <geometry><mesh filename="drawer.obj" scale="1 1 1" /></geometry>
            <origin xyz="0.5 0 0" rpy="0 0 0" />
        </visual>
    </link>
    <link name="door">
        <visual>
            <geometry><mesh filename="door.obj" scale="1 1 1" /></geometry>
            <origin xyz="0.4 0 0" rpy="0 0 0" />
        </visual>
    </link>
    <joint name="drawer_slide" type="prismatic">
        <parent link="base" />
        <child link="drawer" />
        <axis xyz="1 0 0" />
        <origin xyz="0 0 0" rpy="0 0 0" />
        <limit lower="0" upper="0.6" />
    </joint>
    <joint name="door_hinge" type="revolute">
        <parent link="base" />
        <child link="door" />
        <axis xyz="0 0 1" />
        <origin xyz="0 0 0" rpy="0 0 0" />
        <limit lower="0" upper="1.5707963" />
    </joint>
</robot>
"""

# Equivalent MJCF fixture: meshes live in a ``meshes/`` dir referenced by
# ``compiler meshdir``; the drawer slides along +x, the door hinges about
# +z through the door body origin, both with the slab offset from the
# joint anchor (as on a real cabinet).
_DRAWER_DOOR_MJCF = """\
<mujoco model="unit_drawer_door">
    <compiler meshdir="meshes" />
    <asset>
        <mesh name="base_mesh" file="base.obj" />
        <mesh name="drawer_mesh" file="drawer.obj" />
        <mesh name="door_mesh" file="door.obj" />
    </asset>
    <worldbody>
        <body name="base" pos="0 0 0">
            <geom name="base_geom" type="mesh" mesh="base_mesh" />
            <body name="drawer" pos="0.5 0 -0.05">
                <joint name="drawer_slide" type="slide" axis="1 0 0" />
                <geom name="drawer_geom" type="mesh" mesh="drawer_mesh" />
            </body>
            <body name="door" pos="0.4 0 0.1">
                <joint name="door_hinge" type="hinge" axis="0 0 1" />
                <geom name="door_geom" type="mesh" mesh="door_mesh" pos="0.15 0 0" />
            </body>
        </body>
    </worldbody>
</mujoco>
"""


# A URDF whose moving link's FRAME is offset AND rotated from the base
# (joint ``<origin>`` ≠ identity): ``base_T_link(ref)`` is non-trivial, so a
# base-frame bbox combined with ``world_T_link`` is wrong by exactly that
# transform — the regression that the coincident-frame fixtures above cannot
# catch.  The handle's visual mesh additionally carries an in-link offset.
_OFFSET_LINK_URDF = """\
<robot name="offset_link">
    <link name="base">
        <visual>
            <geometry><mesh filename="base.obj" scale="1 1 1" /></geometry>
        </visual>
    </link>
    <link name="handle">
        <visual>
            <geometry><mesh filename="handle.obj" scale="1 1 1" /></geometry>
            <origin xyz="0.1 0 0" rpy="0 0 0" />
        </visual>
    </link>
    <joint name="handle_slide" type="prismatic">
        <parent link="base" />
        <child link="handle" />
        <axis xyz="1 0 0" />
        <origin xyz="0.3 0.2 0" rpy="0 0 1.5707963267948966" />
        <limit lower="0" upper="0.6" />
    </joint>
</robot>
"""


def _write_box_mesh(path: Path, size: float = 0.1) -> None:
    """Write a centered axis-aligned box mesh (OBJ) to *path*."""
    import trimesh

    trimesh.creation.box(extents=(size, size, size)).export(path)


@pytest.fixture
def urdf_dir(tmp_path: Path) -> Path:
    """Directory with the synthetic two-link URDF and its box meshes."""
    (tmp_path / "unit.urdf").write_text(_DRAWER_DOOR_URDF)
    _write_box_mesh(tmp_path / "base.obj", size=0.2)
    _write_box_mesh(tmp_path / "drawer.obj")
    _write_box_mesh(tmp_path / "door.obj")
    return tmp_path


@pytest.fixture
def mjcf_file(tmp_path: Path) -> Path:
    """Synthetic MJCF with a ``compiler meshdir`` asset directory."""
    (tmp_path / "meshes").mkdir()
    _write_box_mesh(tmp_path / "meshes" / "base.obj", size=0.2)
    _write_box_mesh(tmp_path / "meshes" / "drawer.obj")
    _write_box_mesh(tmp_path / "meshes" / "door.obj")
    path = tmp_path / "unit.xml"
    path.write_text(_DRAWER_DOOR_MJCF)
    return path


@pytest.fixture
def offset_urdf_dir(tmp_path: Path) -> Path:
    """Directory with the offset/rotated-link URDF and its box meshes.

    Expected frames (``base_T_handle(ref) = T(0.3, 0.2, 0) · Rz(π/2)``):

    * link-local handle bbox: ``[0.05, 0.15] × [−0.05, 0.05]²`` (the
      authored ``<visual><origin xyz="0.1 0 0">`` around a 0.1 box);
    * base-frame handle bbox at the reference configuration:
      ``[0.25, 0.35] × [0.25, 0.35] × [−0.05, 0.05]`` (the 90° yaw maps the
      link's +x offset onto +y; the AABB re-fit stays exact because the
      rotation is axis-aligned);
    * at ``q = 0.3`` the link frame translates to ``(0.3, 0.5, 0)`` (the
      joint axis is the link's own +x, rotated into world/base +y).
    """
    (tmp_path / "offset.urdf").write_text(_OFFSET_LINK_URDF)
    _write_box_mesh(tmp_path / "base.obj", size=0.2)
    _write_box_mesh(tmp_path / "handle.obj")
    return tmp_path


# ------------------------------------------------------------------
# Lightweight fake entity (no Genesis build)
# ------------------------------------------------------------------

class _FakeJoint:
    def __init__(self, name: str, dofs_idx_local):
        self.name = name
        self.dofs_idx_local = list(dofs_idx_local)


class _FakeLink:
    def __init__(self, name: str, idx_local: int):
        self.name = name
        self.idx_local = idx_local


class _FakeEntity:
    """Offline stand-in for a built ``genesis`` ``RigidEntity``.

    Implements exactly the call signatures the descriptor uses (position /
    limit / link-pose queries, each ``(…,)`` or ``(B, …)`` per env, with an
    optional boolean ``envs_idx`` mask) and records writes so tests can
    assert on them.

    DoF limits follow real Genesis semantics: with the default
    ``batch_dofs_info=False`` they are a **non-batched** ``(n_dofs,)``
    table shared by all envs, and passing ``envs_idx`` raises
    (``GenesisException`` in Genesis).  With ``batch_dofs_info=True`` the
    table is ``(B, n_dofs)`` and ``envs_idx`` selects rows.
    """

    def __init__(
        self,
        joints,
        links,
        qpos: torch.Tensor,
        q_min: torch.Tensor,
        q_max: torch.Tensor,
        link_pos: torch.Tensor,
        link_quat: torch.Tensor,
        batch_dofs_info: bool = False,
    ):
        self.joints = joints
        self.links = links
        self.qpos = torch.as_tensor(qpos, dtype=torch.float32)
        self.q_min = torch.as_tensor(q_min, dtype=torch.float32)
        self.q_max = torch.as_tensor(q_max, dtype=torch.float32)
        self.link_pos = torch.as_tensor(link_pos, dtype=torch.float32)
        self.link_quat = torch.as_tensor(link_quat, dtype=torch.float32)
        self.batch_dofs_info = batch_dofs_info
        self.set_dofs_position_calls = []
        self.set_pos_calls = []
        self.set_quat_calls = []
        self.get_dofs_limit_calls = []

    # -- helpers -----------------------------------------------------
    @staticmethod
    def _select(data: torch.Tensor, idx, envs_idx):
        out = data
        if idx is not None:
            out = out[idx] if out.ndim == 1 else out[:, idx]
        if envs_idx is not None:
            out = out[envs_idx]
        return out

    @staticmethod
    def _select_links(data: torch.Tensor, idx, envs_idx):
        """Genesis link tensors are ``(n_links, C)`` or ``(B, n_links, C)``."""
        out = data
        if idx is not None:
            out = out[idx] if out.ndim == 2 else out[:, idx]
        if envs_idx is not None:
            out = out[envs_idx]
        return out

    # -- DoF state ---------------------------------------------------
    def get_dofs_position(self, dofs_idx_local=None, envs_idx=None):
        return self._select(self.qpos, dofs_idx_local, envs_idx)

    def get_dofs_limit(self, dofs_idx_local=None, envs_idx=None):
        self.get_dofs_limit_calls.append((dofs_idx_local, envs_idx))
        if envs_idx is not None and not self.batch_dofs_info:
            # Mirrors genesis/engine/solvers/kinematic_solver.py:1254.
            raise RuntimeError(
                "`envs_idx` cannot be specified for non-batched dofs info."
            )
        if self.batch_dofs_info:
            return (
                self._select(self.q_min, dofs_idx_local, envs_idx),
                self._select(self.q_max, dofs_idx_local, envs_idx),
            )
        # Non-batched DOF info: a single canonical (n_dofs,) table shared by
        # every env — regardless of whether the scene is batched.
        q_min = self.q_min[0] if self.q_min.ndim > 1 else self.q_min
        q_max = self.q_max[0] if self.q_max.ndim > 1 else self.q_max
        return (
            self._select(q_min, dofs_idx_local, None),
            self._select(q_max, dofs_idx_local, None),
        )

    def set_dofs_position(self, position, dofs_idx_local=None, envs_idx=None, zero_velocity=False):
        position = torch.as_tensor(position, dtype=self.qpos.dtype)
        dofs = list(range(self.qpos.shape[-1])) if dofs_idx_local is None else list(dofs_idx_local)
        self.set_dofs_position_calls.append((position.clone(), dofs, envs_idx, zero_velocity))
        dof_t = torch.as_tensor(dofs, dtype=torch.long)
        if self.qpos.ndim == 1:
            self.qpos[dof_t] = position.reshape(-1)
            return
        rows = (
            torch.nonzero(envs_idx, as_tuple=False).squeeze(-1)
            if envs_idx is not None
            else torch.arange(self.qpos.shape[0])
        )
        self.qpos[rows[:, None], dof_t[None, :]] = position.reshape(len(rows), -1)

    # -- link poses --------------------------------------------------
    def get_links_pos(self, links_idx_local=None, envs_idx=None, relative=True):
        assert relative
        return self._select_links(self.link_pos, links_idx_local, envs_idx)

    def get_links_quat(self, links_idx_local=None, envs_idx=None, relative=True):
        assert relative
        return self._select_links(self.link_quat, links_idx_local, envs_idx)

    # -- base placement (set_pos_3d) ---------------------------------
    def set_pos(self, position, envs_idx=None):
        self.set_pos_calls.append((torch.as_tensor(position).clone(), envs_idx))

    def set_quat(self, quat, envs_idx=None):
        self.set_quat_calls.append((torch.as_tensor(quat).clone(), envs_idx))


def _drawer_door_entity(
    batch_size: int | None = None,
    batch_dofs_info: bool = False,
) -> _FakeEntity:
    """Fake entity: joints ``drawer_slide`` (dof 0), ``door_hinge`` (dof 1),
    links ``base`` (0), ``drawer`` (1), ``door`` (2).

    *batch_dofs_info* mirrors ``RigidOptions(batch_dofs_info=…)`` — the
    default ``False`` means one ``(n_dofs,)`` limit table shared by all envs.
    """
    joints = [
        _FakeJoint("drawer_slide", [0]),
        _FakeJoint("door_hinge", [1]),
    ]
    links = [
        _FakeLink("base", 0),
        _FakeLink("drawer", 1),
        _FakeLink("door", 2),
    ]
    shape = (batch_size, 2) if batch_size is not None else (2,)
    link_shape = (batch_size, 3, 3) if batch_size is not None else (3, 3)
    link_pos = torch.zeros(link_shape)
    link_quat = torch.zeros((batch_size, 3, 4) if batch_size is not None else (3, 4))
    link_quat[..., 0] = 1.0
    return _FakeEntity(
        joints=joints,
        links=links,
        qpos=torch.zeros(shape),
        q_min=torch.tensor([0.0, 0.0]).expand(shape).clone(),
        q_max=torch.tensor([0.6, 1.5]).expand(shape).clone(),
        link_pos=link_pos,
        link_quat=link_quat,
        batch_dofs_info=batch_dofs_info,
    )


def _offset_link_entity(link_pos, link_quat) -> _FakeEntity:
    """Fake entity for ``_OFFSET_LINK_URDF`` at a *known current* link pose.

    Joints ``handle_slide`` (dof 0); links ``base`` (0), ``handle`` (1).
    The caller supplies the handle's current world pose — the FK-consistent
    poses of that URDF are ``T(0.3, 0.2 + q, 0) · Rz(π/2)`` (the joint axis
    is the link's own +x, which the 90° link yaw maps onto +y).
    """
    return _FakeEntity(
        joints=[_FakeJoint("handle_slide", [0])],
        links=[_FakeLink("base", 0), _FakeLink("handle", 1)],
        qpos=torch.zeros(1),
        q_min=torch.tensor([0.0]),
        q_max=torch.tensor([0.6]),
        link_pos=torch.as_tensor(link_pos, dtype=torch.float32).reshape(2, 3),
        link_quat=torch.as_tensor(link_quat, dtype=torch.float32).reshape(2, 4),
    )


class _TestURDFObject(ArticulatedObject):
    """Minimal concrete subclass wiring the offline URDF FK helper.

    Test-only double: ``load_into_scene`` is never needed because the tests
    stay offline; the hub subclasses (later stages) provide the real one.
    """

    def __init__(self, urdf_file: Path | None = None, **kwargs):
        super().__init__(**kwargs)
        self.urdf_file = None if urdf_file is None else str(urdf_file)

    def _load_link_meshes_at_reference(self):
        assert self.urdf_file is not None, "test double configured without an asset"
        return urdf_link_meshes_at_config(self.urdf_file, self.reference_qpos)

    def _load_link_local_meshes_at_reference(self):
        assert self.urdf_file is not None, "test double configured without an asset"
        return urdf_link_meshes_at_config(
            self.urdf_file, self.reference_qpos, link_local=True,
        )

    def load_into_scene(self, world, pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0),
                        name_override=None, batch_fixed_verts=False):
        raise NotImplementedError("offline test double")


# ------------------------------------------------------------------
# Runtime resolvers
# ------------------------------------------------------------------

def test_articulation_error_is_runtime_error():
    assert issubclass(ArticulationError, RuntimeError)


def test_resolve_joint_dofs_success():
    entity = _drawer_door_entity()
    assert resolve_joint_dofs(entity, "drawer_slide") == [0]
    assert resolve_joint_dofs(entity, "door_hinge") == [1]


def test_resolve_joint_dofs_missing_lists_available_names():
    entity = _drawer_door_entity()
    with pytest.raises(ArticulationError, match="no_such_joint"):
        resolve_joint_dofs(entity, "no_such_joint")
    with pytest.raises(ArticulationError) as excinfo:
        resolve_joint_dofs(entity, "no_such_joint")
    assert "drawer_slide" in str(excinfo.value)
    assert "door_hinge" in str(excinfo.value)


def test_resolve_link_index_success():
    entity = _drawer_door_entity()
    assert resolve_link_index(entity, "base") == 0
    assert resolve_link_index(entity, "drawer") == 1
    assert resolve_link_index(entity, "door") == 2


def test_resolve_link_index_missing_lists_available_names():
    entity = _drawer_door_entity()
    with pytest.raises(ArticulationError) as excinfo:
        resolve_link_index(entity, "handle")
    message = str(excinfo.value)
    assert "handle" in message
    assert "drawer" in message and "door" in message


# ------------------------------------------------------------------
# Offline URDF FK
# ------------------------------------------------------------------

def test_urdf_link_meshes_at_config_places_links(urdf_dir):
    zero = urdf_link_meshes_at_config(urdf_dir / "unit.urdf")
    assert set(zero) == {"base", "drawer", "door"}
    # Base visual mesh: 0.2 box at the origin.
    np.testing.assert_allclose(zero["base"].bounds, [[-0.1] * 3, [0.1] * 3], atol=1e-6)
    # Drawer mesh: 0.1 box, offset 0.5 along +x in the (closed) link frame.
    np.testing.assert_allclose(
        zero["drawer"].bounds, [[0.45, -0.05, -0.05], [0.55, 0.05, 0.05]], atol=1e-6,
    )
    # Door mesh: 0.1 box, offset 0.4 along +x.
    np.testing.assert_allclose(
        zero["door"].bounds, [[0.35, -0.05, -0.05], [0.45, 0.05, 0.05]], atol=1e-6,
    )

    open_cfg = urdf_link_meshes_at_config(
        urdf_dir / "unit.urdf", {"drawer_slide": 0.3, "door_hinge": math.pi / 2},
    )
    # Drawer translated 0.3 along +x (its mesh offset rides along).
    np.testing.assert_allclose(
        open_cfg["drawer"].bounds, [[0.75, -0.05, -0.05], [0.85, 0.05, 0.05]], atol=1e-6,
    )
    # Door rotated 90 deg about +z: its +x mesh offset now points along +y.
    np.testing.assert_allclose(
        open_cfg["door"].bounds, [[-0.05, 0.35, -0.05], [0.05, 0.45, 0.05]], atol=1e-6,
    )
    # The base link is unaffected by the joints.
    np.testing.assert_allclose(open_cfg["base"].bounds, zero["base"].bounds, atol=1e-6)


def test_urdf_link_meshes_applies_mesh_scale(tmp_path):
    (tmp_path / "scaled.urdf").write_text(
        """\
<robot name="scaled">
    <link name="base">
        <visual>
            <geometry><mesh filename="box.obj" scale="2 2 2" /></geometry>
        </visual>
    </link>
</robot>
"""
    )
    _write_box_mesh(tmp_path / "box.obj")
    meshes = urdf_link_meshes_at_config(tmp_path / "scaled.urdf")
    np.testing.assert_allclose(meshes["base"].bounds, [[-0.1] * 3, [0.1] * 3], atol=1e-6)


def test_urdf_link_meshes_unknown_joint_raises(urdf_dir):
    with pytest.raises(ArticulationError) as excinfo:
        urdf_link_meshes_at_config(urdf_dir / "unit.urdf", {"nope": 1.0})
    assert "nope" in str(excinfo.value)
    assert "drawer_slide" in str(excinfo.value)


def test_urdf_link_local_meshes_skip_fk(offset_urdf_dir):
    """``link_local=True`` returns the raw visual geometry per link frame."""
    local = urdf_link_meshes_at_config(offset_urdf_dir / "offset.urdf", link_local=True)
    assert set(local) == {"base", "handle"}
    np.testing.assert_allclose(local["base"].bounds, [[-0.1] * 3, [0.1] * 3], atol=1e-6)
    # The handle mesh keeps its authored in-link visual offset: expressed in
    # the LINK's own frame, NOT the base frame (whose FK placement is x≈0.3).
    np.testing.assert_allclose(
        local["handle"].bounds, [[0.05, -0.05, -0.05], [0.15, 0.05, 0.05]], atol=1e-6,
    )
    # Link-local geometry is configuration-independent: a moved joint leaves
    # it untouched (the joint motion belongs to ``world_T_link`` instead).
    moved = urdf_link_meshes_at_config(
        offset_urdf_dir / "offset.urdf", {"handle_slide": 0.3}, link_local=True,
    )
    np.testing.assert_allclose(moved["handle"].bounds, local["handle"].bounds, atol=1e-6)

    # The base-frame mode is unchanged by the new keyword.
    base_frame = urdf_link_meshes_at_config(offset_urdf_dir / "offset.urdf")
    np.testing.assert_allclose(
        base_frame["handle"].bounds, [[0.25, 0.25, -0.05], [0.35, 0.35, 0.05]], atol=1e-6,
    )


# ------------------------------------------------------------------
# Offline MJCF FK
# ------------------------------------------------------------------

def test_mjcf_link_meshes_at_config_places_bodies(mjcf_file):
    zero = mjcf_link_meshes_at_config(mjcf_file)
    assert set(zero) == {"base", "drawer", "door"}
    np.testing.assert_allclose(zero["base"].bounds, [[-0.1] * 3, [0.1] * 3], atol=1e-6)
    # Drawer body at +0.5x, -0.05z; hinge/slide at rest.
    np.testing.assert_allclose(
        zero["drawer"].bounds,
        [[0.45, -0.05, -0.1], [0.55, 0.05, 0.0]],
        atol=1e-6,
    )
    # Door body at +0.4x, +0.1z with the slab offset +0.15x in the body frame.
    np.testing.assert_allclose(
        zero["door"].bounds,
        [[0.5, -0.05, 0.05], [0.6, 0.05, 0.15]],
        atol=1e-6,
    )

    open_cfg = mjcf_link_meshes_at_config(
        mjcf_file, {"drawer_slide": 0.3, "door_hinge": math.pi / 2},
    )
    # Slide: translation along the joint axis (+x).
    np.testing.assert_allclose(
        open_cfg["drawer"].bounds,
        [[0.75, -0.05, -0.1], [0.85, 0.05, 0.0]],
        atol=1e-6,
    )
    # Hinge about +z through the body origin: the slab (offset +0.15x) swings
    # to +y about the door body position (0.4, 0, 0.1).
    np.testing.assert_allclose(
        open_cfg["door"].bounds,
        [[0.35, 0.1, 0.05], [0.45, 0.2, 0.15]],
        atol=1e-6,
    )


def test_mjcf_link_local_meshes_skip_fk(mjcf_file):
    """``link_local=True`` returns the raw geom geometry per body frame."""
    local = mjcf_link_meshes_at_config(mjcf_file, link_local=True)
    assert set(local) == {"base", "drawer", "door"}
    # The drawer geom has no own offset: its body pos (0.5, 0, −0.05) is
    # skipped, so the mesh sits centred on the body frame origin.
    np.testing.assert_allclose(
        local["drawer"].bounds, [[-0.05] * 3, [0.05] * 3], atol=1e-6,
    )
    # The door slab keeps its authored in-body offset (0.15, 0, 0) — the
    # world/base-frame walker places it at x ≈ 0.55 instead.
    np.testing.assert_allclose(
        local["door"].bounds, [[0.10, -0.05, -0.05], [0.20, 0.05, 0.05]], atol=1e-6,
    )
    # Configuration-independent under a moved joint.
    moved = mjcf_link_meshes_at_config(mjcf_file, {"door_hinge": math.pi / 2}, link_local=True)
    np.testing.assert_allclose(moved["door"].bounds, local["door"].bounds, atol=1e-6)


def test_mjcf_body_euler_is_degrees(tmp_path):
    (tmp_path / "euler.xml").write_text(
        """\
<mujoco model="euler_unit">
    <asset>
        <mesh name="box_mesh" file="box.obj" />
    </asset>
    <worldbody>
        <body name="rotated" pos="1 0 0" euler="0 0 90">
            <geom name="box_geom" type="mesh" mesh="box_mesh" pos="0.5 0 0" />
        </body>
    </worldbody>
</mujoco>
"""
    )
    _write_box_mesh(tmp_path / "box.obj")
    meshes = mjcf_link_meshes_at_config(tmp_path / "euler.xml")
    # euler is in degrees by default: the +x offset becomes +y.
    np.testing.assert_allclose(
        meshes["rotated"].bounds,
        [[0.95, 0.45, -0.05], [1.05, 0.55, 0.05]],
        atol=1e-6,
    )


def test_mjcf_unknown_mesh_raises(tmp_path):
    (tmp_path / "broken.xml").write_text(
        """\
<mujoco model="broken">
    <worldbody>
        <body name="base">
            <geom type="mesh" mesh="missing_mesh" />
        </body>
    </worldbody>
</mujoco>
"""
    )
    with pytest.raises(ArticulationError, match="missing_mesh"):
        mjcf_link_meshes_at_config(tmp_path / "broken.xml")


def test_mjcf_unsupported_joint_type_raises(tmp_path):
    (tmp_path / "ball.xml").write_text(
        """\
<mujoco model="ball_unit">
    <worldbody>
        <body name="base">
            <joint name="ball_joint" type="ball" />
        </body>
    </worldbody>
</mujoco>
"""
    )
    with pytest.raises(ArticulationError, match="ball"):
        mjcf_link_meshes_at_config(tmp_path / "ball.xml")


# ------------------------------------------------------------------
# Descriptor bounding boxes (offline union + per-link)
# ------------------------------------------------------------------

def test_local_bbox_union_and_caching(urdf_dir):
    obj = _TestURDFObject(
        urdf_dir / "unit.urdf",
        name="unit",
        reference_qpos={"drawer_slide": 0.3},
    )
    bbox = obj.get_local_bounding_box()
    # Union spans the base (±0.1), the open drawer (x <= 0.3 + 0.5 + 0.05)
    # and the closed door (enclosed by the base) at the reference config.
    np.testing.assert_allclose(bbox.low, [-0.1, -0.1, -0.1], atol=1e-6)
    np.testing.assert_allclose(bbox.high, [0.85, 0.1, 0.1], atol=1e-6)
    assert obj.get_local_bounding_box() is bbox  # cached

    drawer_box = obj.get_link_local_bounding_box("drawer")
    np.testing.assert_allclose(drawer_box.low, [0.75, -0.05, -0.05], atol=1e-6)
    np.testing.assert_allclose(drawer_box.high, [0.85, 0.05, 0.05], atol=1e-6)
    assert obj.get_link_local_bounding_box("drawer") is drawer_box  # cached

    base_box = obj.get_link_local_bounding_box("base")
    assert base_box.low[0] == pytest.approx(-0.1)


def test_link_bbox_unknown_link_raises(urdf_dir):
    obj = _TestURDFObject(urdf_dir / "unit.urdf")
    with pytest.raises(ArticulationError) as excinfo:
        obj.get_link_local_bounding_box("handle")
    assert "handle" in str(excinfo.value)
    assert "drawer" in str(excinfo.value)


def test_local_bbox_uses_reference_qpos(urdf_dir):
    closed = _TestURDFObject(urdf_dir / "unit.urdf")
    opened = _TestURDFObject(
        urdf_dir / "unit.urdf", reference_qpos={"drawer_slide": 0.6},
    )
    assert opened.get_local_bounding_box().high[0] == pytest.approx(1.15)
    assert closed.get_local_bounding_box().high[0] == pytest.approx(0.55)


# ------------------------------------------------------------------
# Joint state by name (runtime, entity-backed)
# ------------------------------------------------------------------

def test_get_set_dofs_position_by_name():
    entity = _drawer_door_entity()
    entity.qpos[0] = 0.25
    obj = _TestURDFObject()
    pos = obj.get_dofs_position(entity, "drawer_slide")
    assert pos.shape == (1,)
    assert pos[0].item() == pytest.approx(0.25)

    obj.set_dofs_position(entity, torch.tensor([0.4]), "drawer_slide")
    assert entity.qpos[0].item() == pytest.approx(0.4)
    assert entity.set_dofs_position_calls[-1][1] == [0]
    assert entity.set_dofs_position_calls[-1][3] is True  # zero_velocity default


def test_get_dofs_position_unknown_joint_raises():
    entity = _drawer_door_entity()
    obj = _TestURDFObject()
    with pytest.raises(ArticulationError, match="drawer_slide"):
        obj.get_dofs_position(entity, "no_such_joint")


def test_get_dofs_position_batched_with_mask():
    entity = _drawer_door_entity(batch_size=2)
    entity.qpos[0, 0] = 0.1
    entity.qpos[1, 0] = 0.2
    obj = _TestURDFObject()
    full = obj.get_dofs_position(entity, "drawer_slide")
    assert full.shape == (2, 1)
    np.testing.assert_allclose(full[:, 0].tolist(), [0.1, 0.2])
    masked = obj.get_dofs_position(
        entity, "drawer_slide", envs_idx=torch.tensor([False, True]),
    )
    assert masked.shape == (1, 1)
    assert masked[0, 0].item() == pytest.approx(0.2)


# ------------------------------------------------------------------
# Open fraction
# ------------------------------------------------------------------

@pytest.mark.parametrize(
    "q, expected",
    [(0.0, 0.0), (0.3, 0.5), (0.6, 1.0)],
)
def test_get_open_fraction_math(q, expected):
    entity = _drawer_door_entity()
    entity.qpos[0] = q
    obj = _TestURDFObject()
    fraction = obj.get_open_fraction(entity, "drawer_slide")
    assert fraction.shape == (1,)
    assert fraction[0].item() == pytest.approx(expected)


def test_get_open_fraction_clamps_out_of_range():
    entity = _drawer_door_entity()
    obj = _TestURDFObject()
    entity.qpos[0] = -0.5
    assert obj.get_open_fraction(entity, "drawer_slide")[0].item() == pytest.approx(0.0)
    entity.qpos[0] = 10.0
    assert obj.get_open_fraction(entity, "drawer_slide")[0].item() == pytest.approx(1.0)


def test_get_open_fraction_unbounded_limits_raise():
    entity = _drawer_door_entity()
    entity.q_min[0] = -torch.inf
    entity.q_max[0] = torch.inf
    obj = _TestURDFObject()
    with pytest.raises(ArticulationError, match="drawer_slide"):
        obj.get_open_fraction(entity, "drawer_slide")


def test_get_open_fraction_batched_with_mask():
    entity = _drawer_door_entity(batch_size=2)
    entity.qpos[0, 0] = 0.3   # fraction 0.5
    entity.qpos[1, 0] = 0.6   # fraction 1.0
    obj = _TestURDFObject()
    masked = obj.get_open_fraction(
        entity, "drawer_slide", envs_idx=torch.tensor([True, False]),
    )
    assert masked.shape == (1, 1)
    assert masked[0, 0].item() == pytest.approx(0.5)


def test_set_open_fraction_clamps_and_writes_by_name():
    entity = _drawer_door_entity()
    obj = _TestURDFObject()
    obj.set_open_fraction(entity, "drawer_slide", torch.tensor([0.5]))
    assert entity.qpos[0].item() == pytest.approx(0.3)
    obj.set_open_fraction(entity, "drawer_slide", torch.tensor([1.5]))
    assert entity.qpos[0].item() == pytest.approx(0.6)
    obj.set_open_fraction(entity, "drawer_slide", torch.tensor([-0.2]))
    assert entity.qpos[0].item() == pytest.approx(0.0)
    assert entity.set_dofs_position_calls[-1][1] == [0]


def test_set_open_fraction_batched_and_masked():
    entity = _drawer_door_entity(batch_size=2)
    obj = _TestURDFObject()
    # Per-env fractions (B,) broadcast to the single DOF of the joint.
    obj.set_open_fraction(entity, "drawer_slide", torch.tensor([0.0, 1.0]))
    np.testing.assert_allclose(entity.qpos[:, 0].tolist(), [0.0, 0.6])
    # Masked write touches only the selected env; the joint is addressed by name.
    obj.set_open_fraction(
        entity,
        "drawer_slide",
        torch.tensor([0.5]),
        envs_idx=torch.tensor([True, False]),
    )
    np.testing.assert_allclose(entity.qpos[:, 0].tolist(), [0.3, 0.6])
    assert entity.set_dofs_position_calls[-1][2] is not None


def test_open_fraction_masked_never_forwards_envs_idx_to_limits():
    """Regression: limits are read unmasked even when a mask is supplied.

    With ``RigidOptions(batch_dofs_info=False)`` (the fake's default) real
    Genesis raises ``GenesisException`` when ``envs_idx`` reaches
    ``get_dofs_limit``; limits are scene-wide constants, so the open-fraction
    methods must fetch them without ``envs_idx`` while still masking the
    position read/write.
    """
    entity = _drawer_door_entity(batch_size=2)
    entity.qpos[0, 0] = 0.3   # fraction 0.5
    entity.qpos[1, 0] = 0.6   # fraction 1.0
    obj = _TestURDFObject()
    mask = torch.tensor([True, False])

    fraction = obj.get_open_fraction(entity, "drawer_slide", envs_idx=mask)
    assert fraction.shape == (1, 1)
    assert fraction[0, 0].item() == pytest.approx(0.5)
    dofs_idx, limit_envs_idx = entity.get_dofs_limit_calls[-1]
    assert dofs_idx == [0]
    assert limit_envs_idx is None  # limits fetched unmasked (no Genesis raise)

    obj.set_open_fraction(entity, "drawer_slide", torch.tensor([0.5]), envs_idx=mask)
    dofs_idx, limit_envs_idx = entity.get_dofs_limit_calls[-1]
    assert dofs_idx == [0]
    assert limit_envs_idx is None
    position, dofs, write_envs_idx, _ = entity.set_dofs_position_calls[-1]
    assert write_envs_idx is mask  # the real mask still reaches the write
    assert dofs == [0]
    np.testing.assert_allclose(position.tolist(), [0.3])
    # Only env 0 was written; env 1 keeps its previous position.
    np.testing.assert_allclose(entity.qpos[:, 0].tolist(), [0.3, 0.6])


def test_open_fraction_masked_with_batched_dofs_info():
    """``batch_dofs_info=True``: unmasked limits are per-env, masked locally."""
    entity = _drawer_door_entity(batch_size=2, batch_dofs_info=True)
    entity.q_min[:, 0] = 0.0
    entity.q_max[:, 0] = torch.tensor([0.6, 1.2])  # per-env ranges
    entity.qpos[0, 0] = 0.3   # env 0 fraction 0.5
    entity.qpos[1, 0] = 0.9   # env 1 fraction 0.75
    obj = _TestURDFObject()
    mask = torch.tensor([True, False])

    fraction = obj.get_open_fraction(entity, "drawer_slide", envs_idx=mask)
    assert fraction.shape == (1, 1)
    assert fraction[0, 0].item() == pytest.approx(0.5)
    assert entity.get_dofs_limit_calls[-1][1] is None

    obj.set_open_fraction(entity, "drawer_slide", torch.tensor([1.0]), envs_idx=mask)
    assert entity.get_dofs_limit_calls[-1][1] is None
    # Env 0 saturated at its own q_max, env 1 untouched.
    np.testing.assert_allclose(entity.qpos[:, 0].tolist(), [0.6, 0.9])


# ------------------------------------------------------------------
# Per-link runtime pose
# ------------------------------------------------------------------

def test_get_link_pose_unbatched():
    entity = _drawer_door_entity()
    entity.link_pos[1] = torch.tensor([0.3, 0.0, 0.0])
    entity.link_quat[1] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    obj = _TestURDFObject()
    pos, quat = obj.get_link_pose(entity, "drawer")
    assert pos.shape == (3,) and quat.shape == (4,)
    np.testing.assert_allclose(pos.tolist(), [0.3, 0.0, 0.0])
    np.testing.assert_allclose(quat.tolist(), [0.0, 0.0, 0.0, 1.0])


def test_get_link_pose_batched_and_unknown_link():
    entity = _drawer_door_entity(batch_size=2)
    entity.link_pos[0, 1] = torch.tensor([0.3, 0.0, 0.0])
    entity.link_pos[1, 1] = torch.tensor([0.4, 0.0, 0.0])
    obj = _TestURDFObject()
    pos, quat = obj.get_link_pose(entity, "drawer")
    assert pos.shape == (2, 3) and quat.shape == (2, 4)
    np.testing.assert_allclose(pos[:, 0].tolist(), [0.3, 0.4])
    with pytest.raises(ArticulationError, match="handle"):
        obj.get_link_pose(entity, "handle")


# ------------------------------------------------------------------
# Part-aware grasp candidates
# ------------------------------------------------------------------

def _yaw_rotation(angle: float) -> np.ndarray:
    """Independent 3x3 yaw matrix (test-side reference implementation)."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_part_grasping_candidates_world_frame(urdf_dir):
    obj = _TestURDFObject(urdf_dir / "unit.urdf")
    entity = _drawer_door_entity()
    entity.link_pos[1] = torch.tensor([1.0, 2.0, 0.5])

    candidates = obj.list_part_grasping_candidates(
        entity, "drawer", gripper_width=0.5,
    )
    # 0.1 box: all three axes are narrower than the gripper -> 3 * 2 * 2 * 2.
    assert candidates.position.shape == (24, 3)
    local = obj._part_grasp_cache[("drawer", 0.5, 0.01)]
    np.testing.assert_allclose(
        candidates.position.numpy(),
        local.position.numpy() + np.array([1.0, 2.0, 0.5]),
        atol=1e-5,
    )
    # Directions are unchanged by an identity rotation.
    np.testing.assert_allclose(
        candidates.approach_direction.numpy(), local.approach_direction.numpy(), atol=1e-5,
    )

    # Cached: a second call reuses the same local candidates.
    again = obj.list_part_grasping_candidates(entity, "drawer", gripper_width=0.5)
    np.testing.assert_allclose(again.position.numpy(), candidates.position.numpy(), atol=1e-5)
    assert len(obj._part_grasp_cache) == 1


def test_part_grasping_candidates_track_link_rotation(urdf_dir):
    obj = _TestURDFObject(urdf_dir / "unit.urdf")
    entity = _drawer_door_entity()
    angle = math.pi / 2
    entity.link_pos[2] = torch.tensor([0.0, 0.0, 0.3])
    entity.link_quat[2] = torch.tensor([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])

    candidates = obj.list_part_grasping_candidates(
        entity, "door", gripper_width=0.5,
    )
    local = obj._part_grasp_cache[("door", 0.5, 0.01)]
    expected = local.position.numpy() @ _yaw_rotation(angle).T + np.array([0.0, 0.0, 0.3])
    np.testing.assert_allclose(candidates.position.numpy(), expected, atol=1e-5)
    expected_dirs = local.approach_direction.numpy() @ _yaw_rotation(angle).T
    np.testing.assert_allclose(
        candidates.approach_direction.numpy(), expected_dirs, atol=1e-5,
    )


def test_part_grasping_candidates_sit_on_offset_link_geometry(offset_urdf_dir):
    """Regression: part candidates enumerate in the LINK frame, not the base frame.

    ``base_T_handle(ref) = T(0.3, 0.2, 0) · Rz(π/2)`` here, so a base-frame
    bbox multiplied by ``world_T_link`` double-applies that transform and
    shifts every candidate by the link's offset/rotation from the base.  At
    the reference configuration (the entity's current pose IS the FK pose)
    the candidates must sit on the handle mesh's world AABB.
    """
    obj = _TestURDFObject(offset_urdf_dir / "offset.urdf")
    angle = math.pi / 2
    entity = _offset_link_entity(
        link_pos=[(0.0, 0.0, 0.0), (0.3, 0.2, 0.0)],
        link_quat=[
            (1.0, 0.0, 0.0, 0.0),
            (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)),
        ],
    )

    candidates = obj.list_part_grasping_candidates(entity, "handle", gripper_width=0.5)
    assert candidates.position.shape == (24, 3)
    local = obj._part_grasp_cache[("handle", 0.5, 0.01)]
    # The enumeration is in the LINK frame: centred on the authored visual
    # offset (0.1, 0, 0) — NOT on the base-frame bbox centre (0.3, 0.3, 0).
    np.testing.assert_allclose(
        local.position.mean(dim=0).numpy(), [0.1, 0.0, 0.0], atol=1e-5,
    )
    # ... and the world candidates are exactly link-local ∘ current pose.
    expected = local.position.numpy() @ _yaw_rotation(angle).T + np.array([0.3, 0.2, 0.0])
    np.testing.assert_allclose(candidates.position.numpy(), expected, atol=1e-5)
    expected_dirs = local.approach_direction.numpy() @ _yaw_rotation(angle).T
    np.testing.assert_allclose(
        candidates.approach_direction.numpy(), expected_dirs, atol=1e-5,
    )

    # Every candidate lies on the handle's actual geometry in world.
    pts = candidates.position.numpy()
    assert (pts >= np.array([0.25, 0.25, -0.05]) - 1e-5).all()
    assert (pts <= np.array([0.35, 0.35, 0.05]) + 1e-5).all()

    # The base-frame bbox contract is untouched (union bbox / receptacles).
    base_bbox = obj.get_link_local_bounding_box("handle")
    np.testing.assert_allclose(base_bbox.low.numpy(), [0.25, 0.25, -0.05], atol=1e-6)
    np.testing.assert_allclose(base_bbox.high.numpy(), [0.35, 0.35, 0.05], atol=1e-6)


def test_part_grasping_candidates_track_slide_joint(offset_urdf_dir):
    """Opening the slide moves the candidates by exactly the slide translation.

    The joint axis is the handle link's own +x, rotated onto world +y by the
    link's authored frame — so the world delta between the reference and the
    moved configuration is (0, 0.3, 0) for q = 0.3, and the moved candidates
    sit on the moved geometry (rigid attachment through ``world_T_link``).
    """
    obj = _TestURDFObject(offset_urdf_dir / "offset.urdf")
    angle = math.pi / 2
    link_quat = [
        (1.0, 0.0, 0.0, 0.0),
        (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)),
    ]

    reference = _offset_link_entity(
        link_pos=[(0.0, 0.0, 0.0), (0.3, 0.2, 0.0)], link_quat=link_quat,
    )
    at_reference = obj.list_part_grasping_candidates(reference, "handle", gripper_width=0.5)

    moved = _offset_link_entity(
        link_pos=[(0.0, 0.0, 0.0), (0.3, 0.5, 0.0)], link_quat=link_quat,
    )
    slid = obj.list_part_grasping_candidates(moved, "handle", gripper_width=0.5)
    np.testing.assert_allclose(
        slid.position.numpy() - at_reference.position.numpy(),
        np.tile([0.0, 0.3, 0.0], (len(at_reference), 1)),
        atol=1e-5,
    )

    # The slid candidates sit on the slid handle mesh in world.
    pts = slid.position.numpy()
    assert (pts >= np.array([0.25, 0.55, -0.05]) - 1e-5).all()
    assert (pts <= np.array([0.35, 0.65, 0.05]) + 1e-5).all()


def test_part_grasping_candidates_batched(urdf_dir):
    obj = _TestURDFObject(urdf_dir / "unit.urdf")
    entity = _drawer_door_entity(batch_size=2)
    entity.link_pos[0, 1] = torch.tensor([0.5, 0.0, 0.0])
    entity.link_pos[1, 1] = torch.tensor([0.9, 0.0, 0.0])

    candidates = obj.list_part_grasping_candidates(
        entity, "drawer", gripper_width=0.5,
    )
    assert candidates.position.shape == (2, 24, 3)
    local = obj._part_grasp_cache[("drawer", 0.5, 0.01)]
    np.testing.assert_allclose(
        candidates.position[0].numpy(),
        local.position.numpy() + np.array([0.5, 0.0, 0.0]),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        candidates.position[1].numpy(),
        local.position.numpy() + np.array([0.9, 0.0, 0.0]),
        atol=1e-5,
    )


def test_part_grasping_candidates_masked_envs_idx(urdf_dir):
    """``envs_idx`` selects the env subset used for the world transform."""
    obj = _TestURDFObject(urdf_dir / "unit.urdf")
    entity = _drawer_door_entity(batch_size=2)
    entity.link_pos[0, 1] = torch.tensor([0.5, 0.0, 0.0])
    entity.link_pos[1, 1] = torch.tensor([0.9, 0.0, 0.0])

    mask = torch.tensor([False, True])
    candidates = obj.list_part_grasping_candidates(
        entity, "drawer", gripper_width=0.5, envs_idx=mask,
    )
    assert candidates.position.shape == (1, 24, 3)
    local = obj._part_grasp_cache[("drawer", 0.5, 0.01)]
    np.testing.assert_allclose(
        candidates.position[0].numpy(),
        local.position.numpy() + np.array([0.9, 0.0, 0.0]),
        atol=1e-5,
    )
    # The cached enumeration is untouched by the mask.
    assert len(obj._part_grasp_cache) == 1
    # Without a mask all envs are returned.
    all_envs = obj.list_part_grasping_candidates(entity, "drawer", gripper_width=0.5)
    assert all_envs.position.shape == (2, 24, 3)


def test_part_grasping_candidates_empty_for_degenerate_flat_link(tmp_path):
    """A degenerate (flat / zero-thickness) link bbox yields empty candidates.

    With a zero-opening gripper no axis of the flat plate's bbox is narrower
    than the opening, so the enumeration is empty; the ``(0, 3)`` candidates
    must survive the local→world transform (unbatched, batched and masked).
    """
    import trimesh

    plate = trimesh.creation.box(extents=(0.4, 0.4, 0.0))
    plate.export(tmp_path / "plate.obj")
    (tmp_path / "plate.urdf").write_text(
        """\
<robot name="flat_plate">
    <link name="base">
        <visual>
            <geometry><mesh filename="plate.obj" scale="1 1 1" /></geometry>
        </visual>
    </link>
</robot>
"""
    )
    obj = _TestURDFObject(tmp_path / "plate.urdf")
    link_box = obj.get_link_local_bounding_box("base")
    np.testing.assert_allclose((link_box.high - link_box.low).numpy(), [0.4, 0.4, 0.0], atol=1e-6)

    entity = _drawer_door_entity()
    empty = obj.list_part_grasping_candidates(
        entity, "base", gripper_width=0.0,
    )
    assert empty.position.shape == (0, 3)
    assert empty.approach_direction.shape == (0, 3)
    assert empty.opening_direction.shape == (0, 3)
    assert len(empty) == 0

    batched = _drawer_door_entity(batch_size=2)
    batched_candidates = obj.list_part_grasping_candidates(
        batched, "base", gripper_width=0.0,
    )
    assert batched_candidates.position.shape == (2, 0, 3)
    masked_candidates = obj.list_part_grasping_candidates(
        batched, "base", gripper_width=0.0, envs_idx=torch.tensor([True, False]),
    )
    assert masked_candidates.position.shape == (1, 0, 3)


def test_part_grasping_candidates_alias_and_unknown_link(urdf_dir):
    obj = _TestURDFObject(urdf_dir / "unit.urdf")
    entity = _drawer_door_entity()

    aliased = obj.list_part_grasping_candidates(
        entity, "handle", gripper_width=0.5, part_aliases={"handle": "drawer"},
    )
    assert aliased.position.shape == (24, 3)
    assert ("drawer", 0.5, 0.01) in obj._part_grasp_cache
    assert ("handle", 0.5, 0.01) not in obj._part_grasp_cache

    with pytest.raises(ArticulationError) as excinfo:
        obj.list_part_grasping_candidates(entity, "handle", gripper_width=0.5)
    assert "handle" in str(excinfo.value)
    assert "drawer" in str(excinfo.value)  # available links are listed


# ------------------------------------------------------------------
# Placement
# ------------------------------------------------------------------

def test_set_pos_3d_defaults_to_non_randomised_quaternion(urdf_dir):
    obj = _TestURDFObject(urdf_dir / "unit.urdf")
    entity = _drawer_door_entity()
    obj.set_pos_3d(entity, torch.tensor([1.0, 2.0, 0.0]))
    pos, _ = entity.set_pos_calls[-1]
    quat, _ = entity.set_quat_calls[-1]
    np.testing.assert_allclose(pos.tolist(), [1.0, 2.0, 0.0])
    # Default random_quaternion=False keeps the authored (identity) yaw.
    np.testing.assert_allclose(quat.tolist(), [1.0, 0.0, 0.0, 0.0])
    assert entity.set_pos_calls[-1][1] is None


def test_set_pos_3d_random_quaternion_and_mask():
    obj = _TestURDFObject()
    entity = _drawer_door_entity(batch_size=2)
    obj.set_pos_3d(
        entity,
        torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        random_quaternion=True,
        rng=torch.Generator().manual_seed(0),
    )
    quat, envs_idx = entity.set_quat_calls[-1]
    assert quat.shape == (2, 4)
    assert envs_idx is None
    assert not torch.allclose(quat[:, 0], torch.ones(2))  # yaw randomised

    mask = torch.tensor([True, False])
    obj.set_pos_3d(
        entity,
        torch.tensor([[3.0, 0.0, 0.0]]),
        random_quaternion=False,
        envs_idx=mask,
    )
    pos, envs_idx = entity.set_pos_calls[-1]
    np.testing.assert_allclose(pos.tolist(), [[3.0, 0.0, 0.0]])
    assert envs_idx is mask


# ------------------------------------------------------------------
# Real-Genesis check (offset link: candidates sit on the link's geometry)
# ------------------------------------------------------------------

def _quat_to_matrix_np(quat_wxyz: np.ndarray) -> np.ndarray:
    """Independent quat (w, x, y, z) → 3x3 rotation matrix."""
    from trimesh.transformations import quaternion_matrix

    return np.asarray(quaternion_matrix(quat_wxyz))[:3, :3]


def _real_genesis_offset_link_part_grasps(backend, tmp_path: Path) -> None:
    """Shared body of the real-Genesis offset-link assertions."""
    import genesis as gs

    if getattr(gs, "_initialized", False):
        gs.destroy()
    gs.init(backend=backend)

    from unienv_genesis import GenesisWorld

    (tmp_path / "offset_real.urdf").write_text(_OFFSET_LINK_URDF)
    _write_box_mesh(tmp_path / "base.obj", size=0.2)
    _write_box_mesh(tmp_path / "handle.obj")

    world = GenesisWorld(scene_init_kwargs={"show_viewer": False})
    try:
        entity = world.scene.add_entity(
            gs.morphs.URDF(
                file=str(tmp_path / "offset_real.urdf"),
                pos=(1.0, 2.0, 0.5),
                quat=(1.0, 0.0, 0.0, 0.0),
                align=False,
                fixed=True,
                merge_fixed_links=False,
            ),
            name="offset_link_fixture",
        )
        world.build_scene(world.scene)

        obj = _TestURDFObject(tmp_path / "offset_real.urdf")
        link_idx = resolve_link_index(entity, "handle")
        link = entity.links[link_idx]

        def _assert_on_geometry(points: np.ndarray, current_pos, current_quat) -> None:
            # (1) Inside the link's visual AABB straight from the entity.
            v_aabb = link.get_vAABB().detach().cpu().numpy()
            assert (points >= v_aabb[0] - 1e-3).all()
            assert (points <= v_aabb[1] + 1e-3).all()
            # (2) Centred on the authored visual offset (0.1, 0, 0) expressed
            # through the *entity's* link pose — the buggy version combined a
            # base-frame bbox with the link pose instead, shifting the
            # candidates by (−0.3, −0.2) in the link's rotated frame and off
            # the geometry.
            rotation = _quat_to_matrix_np(current_quat.detach().cpu().numpy())
            expected_center = (
                current_pos.detach().cpu().numpy() + rotation @ np.array([0.1, 0.0, 0.0])
            )
            np.testing.assert_allclose(points.mean(axis=0), expected_center, atol=1e-3)

        # Reference configuration (all joints at their default positions).
        pos, quat = obj.get_link_pose(entity, "handle")
        candidates = obj.list_part_grasping_candidates(entity, "handle", gripper_width=0.5)
        pts = candidates.position.detach().cpu().numpy()
        assert pts.shape[-1] == 3
        _assert_on_geometry(pts, pos, quat)

        # Slide the joint fully open: the candidates move with the link
        # (rigid attachment) and stay on the moved geometry.  The joint is
        # prismatic, so the per-candidate delta is exactly the link pose's
        # translation delta.
        obj.set_open_fraction(entity, "handle_slide", torch.tensor([1.0]))
        pos_open, quat_open = obj.get_link_pose(entity, "handle")
        candidates_open = obj.list_part_grasping_candidates(entity, "handle", gripper_width=0.5)
        pts_open = candidates_open.position.detach().cpu().numpy()
        assert float(np.abs(quat_open.detach().cpu().numpy() - quat.detach().cpu().numpy()).max()) < 1e-5
        np.testing.assert_allclose(
            pts_open - pts,
            np.tile(
                (pos_open - pos).detach().cpu().numpy(),
                (len(pts), 1),
            ),
            atol=1e-4,
        )
        _assert_on_geometry(pts_open, pos_open, quat_open)
    finally:
        world.scene.destroy()
        # Leave a healthy (CPU) Genesis state for the rest of the suite —
        # the same convention as the real-scene tests in
        # test_open_fixture_task.py (destroying without re-initialising
        # would break modules whose offline fakes still need `gs.init`).
        if getattr(gs, "_initialized", False):
            gs.destroy()
        gs.init(backend=gs.cpu)


@pytest.mark.parametrize(
    "backend_name",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is required",
            ),
        ),
    ],
)
def test_part_grasping_candidates_real_genesis_offset_link(backend_name, tmp_path):
    """Real Genesis entity: candidates sit on the offset link's geometry.

    The link frame is offset AND rotated from the base (``base_T_link(ref) =
    T(0.3, 0.2, 0) · Rz(π/2)``), so this fails against the old behaviour,
    which enumerated the base-frame bbox and multiplied it by
    ``world_T_link`` — double-applying the base→link transform.
    """
    import genesis as gs

    _real_genesis_offset_link_part_grasps(
        gs.cpu if backend_name == "cpu" else gs.cuda, tmp_path,
    )
