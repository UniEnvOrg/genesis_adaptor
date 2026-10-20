"""Tests for CameraSensorNode observation shapes and intrinsics/extrinsics (unbatched, CPU)."""

from __future__ import annotations

import sys

import math
import pytest
import torch
import genesis as gs
from genesis.constants import backend as gs_backend

if not gs._initialized:
    gs.init(backend=gs_backend.cpu)

from unienv_interface.world import WorldEnv
from unienv_interface.world.node import WorldNode
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from xbarray.transformations.rotation_conversions import quaternion_to_matrix

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.sensors import (
    CameraSensorNode,
    CameraSensorNodeOptions,
)
from unienv_genesis.options.attach_option import AttachFromName

# Check if NYX is available — **lazily**, through the camera node's helper.
# ``gs_nyx`` bundles its own monolithic OpenUSD runtime
# (``gs_nyx/libusd_ms.so``), which conflicts with the ``usd-core`` one the
# ArtVIP/USD tests use (Tf notice registry corruption in one process — see
# camera_node.py's DSO note).  Importing the Nyx DSOs at *collection* time
# would poison every other module of the run; the check is deferred to the
# module-level ``__getattr__`` hook only when the Nyx tests actually run.
from unienv_genesis.nodes.sensors import camera_node as _camera_node_module

NYX_AVAILABLE = None  # evaluated lazily (see _nyx_available)


def _nyx_available():
    if NYX_AVAILABLE is None:
        # The Nyx DSOs corrupt the usd-core Tf runtime when a pxr stage was
        # already opened in the same process (ArtVIP tests import pxr lazily
        # at call time); Nyx tests skip rather than crash the whole run then.
        if "pxr" in sys.modules:
            return False
        globals()["_camera_node_state"] = _camera_node_module._NYX_STATE
        globals()["NYX_AVAILABLE"] = _camera_node_module._ensure_nyx() is True
        globals()["npr"] = _camera_node_module._NYX_STATE["npr"]
        globals()["nps"] = _camera_node_module._NYX_STATE["nps"]
        globals()["NyxCameraOptions"] = _camera_node_module._NYX_STATE["NyxCameraOptions"]
    return NYX_AVAILABLE


class DummyBoxNode(WorldNode):
    """Static box entity so the scene is non-empty."""

    reload_priorities = {20}
    reset_priorities = {20}
    pre_environment_step_priorities = {20}
    post_environment_step_priorities = {20}

    def __init__(self, world: GenesisWorld, name: str = "box"):
        self.world = world
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def reload(self, *, priority=0, **kw):
        if priority == 20:
            self.world.scene.add_entity(
                gs.morphs.Box(size=(0.1, 0.1, 0.1), fixed=True, pos=(0, 0, 0.2)),
                name=self._name,
            )

    def reset(self, *, priority=0, **kw):
        pass

    def pre_environment_step(self, dt, *, priority=0):
        pass

    def post_environment_step(self, dt, *, priority=0):
        pass

    def get_observation(self):
        return None


class MovableBoxNode(WorldNode):
    """Free (non-fixed) box entity whose pose can be changed at runtime."""

    reload_priorities = {20}
    reset_priorities = {20}
    pre_environment_step_priorities = {20}
    post_environment_step_priorities = {20}

    def __init__(self, world: GenesisWorld, name: str = "box", pos=(0.0, 0.0, 0.2)):
        self.world = world
        self._name = name
        self._pos = pos

    @property
    def name(self) -> str:
        return self._name

    def reload(self, *, priority=0, **kw):
        if priority == 20:
            self.world.scene.add_entity(
                gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=self._pos),
                name=self._name,
            )

    def reset(self, *, priority=0, **kw):
        pass

    def pre_environment_step(self, dt, *, priority=0):
        pass

    def post_environment_step(self, dt, *, priority=0):
        pass

    def get_observation(self):
        return None


RES = (64, 64)
CAM_POS = (1.0, 0.0, 0.5)
CAM_LOOKAT = (0.0, 0.0, 0.2)


def _cam_opt(**overrides) -> CameraSensorNodeOptions:
    kw = dict(
        res=RES,
        pos=CAM_POS,
        quat=CameraSensorNodeOptions.quat_from_lookat(CAM_POS, CAM_LOOKAT),
        fov=45,
        GUI=False,
    )
    kw.update(overrides)
    return CameraSensorNodeOptions(**kw)


def _make_unbatched_env():
    world = GenesisWorld()
    box = DummyBoxNode(world)
    camera = CameraSensorNode(
        world=world,
        name="cam",
        option=_cam_opt(),
        render_rgb=True,
        render_depth=True,
        render_seg=True,
        render_position=True,
    )
    env = WorldEnv(world, [box, camera], render_mode=None)
    return env, camera


def test_unbatched_observation_shapes():
    env, camera = _make_unbatched_env()
    env.reset(seed=0, reload=True)
    cam_obs = camera.get_observation()

    W, H = RES
    assert cam_obs["rgb"].shape == (H, W, 3)
    assert cam_obs["depth"].shape == (H, W)
    assert cam_obs["seg"].shape == (H, W)
    assert cam_obs["position"].shape == (H, W, 3)
    assert cam_obs["intrinsics_cv"].shape == (3, 3)
    assert cam_obs["extrinsics_cv"].shape == (3, 4)
    env.close()


def test_unbatched_intrinsics_values():
    env, camera = _make_unbatched_env()
    env.reset(seed=0, reload=True)
    K = camera.get_observation()["intrinsics_cv"]

    assert K.shape == (3, 3)
    # fx == fy (single focal length in Genesis)
    assert K[0, 0] == K[1, 1]
    # Bottom row is [0, 0, 1]
    assert K[2, 0] == 0.0
    assert K[2, 1] == 0.0
    assert K[2, 2] == 1.0
    # Skew is zero
    assert K[0, 1] == 0.0
    assert K[1, 0] == 0.0
    env.close()


def test_unbatched_extrinsics_values():
    env, camera = _make_unbatched_env()
    env.reset(seed=0, reload=True)
    E = camera.get_observation()["extrinsics_cv"]

    assert E.shape == (3, 4)
    R = E[:, :3]
    # R should be a valid rotation: R @ R^T ~= I, det(R) ~= 1
    assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-4)
    assert torch.allclose(torch.det(R), torch.tensor(1.0), atol=1e-4)
    env.close()


def test_unbatched_observation_space_matches():
    """Observation space shape should match actual observation shapes."""
    env, camera = _make_unbatched_env()
    env.reset(seed=0, reload=True)
    cam_obs = camera.get_observation()

    for key, space in camera.observation_space.spaces.items():
        assert key in cam_obs, f"Key {key!r} in obs space but not in obs"
        assert cam_obs[key].shape == space.shape, (
            f"Shape mismatch for {key!r}: obs={cam_obs[key].shape}, space={space.shape}"
        )
        # The Rasterizer returns int64 segmentation; the space declares int32.
        assert cam_obs[key].dtype == space.dtype, (
            f"Dtype mismatch for {key!r}: obs={cam_obs[key].dtype}, space={space.dtype}"
        )
    env.close()


def _make_rasterized_env(batch_size):
    """A batched world rendered by the (non-batched) Rasterizer."""
    world = GenesisWorld(
        batch_size=batch_size,
        scene_init_kwargs={"renderer": gs.renderers.Rasterizer()},
    )
    box = DummyBoxNode(world)
    camera = CameraSensorNode(
        world=world,
        name="cam",
        option=_cam_opt(),
        render_rgb=True,
        render_depth=True,
        render_seg=True,
        render_position=True,
    )
    env = WorldEnv(world, [box, camera], render_mode=None)
    return env, camera


def _assert_matches_space(camera, cam_obs):
    for key, space in camera.observation_space.spaces.items():
        assert cam_obs[key].shape == space.shape, (key, cam_obs[key].shape, space.shape)
        assert cam_obs[key].dtype == space.dtype, (key, cam_obs[key].dtype, space.dtype)


def test_rasterized_batch_of_one_keeps_its_batch_dimension():
    """The Rasterizer renders one environment without a batch dimension; a
    batched world of one still gets ``(1, ...)`` leaves, through reset, step
    and a masked reset (which slices the observation by the mask)."""
    env, camera = _make_rasterized_env(batch_size=1)
    env.reset(seed=0, reload=True)
    W, H = RES
    cam_obs = camera.get_observation()
    assert cam_obs["rgb"].shape == (1, H, W, 3)
    assert cam_obs["extrinsics_cv"].shape == (1, 3, 4)
    _assert_matches_space(camera, cam_obs)

    env.step(None)
    _assert_matches_space(camera, camera.get_observation())

    _, obs, _ = env.reset(mask=torch.ones(1, dtype=torch.bool, device=camera.world.device))
    assert obs["depth"].shape == (1, H, W)
    env.close()


def test_rasterized_batch_of_several_is_rejected():
    """A non-batched renderer cannot render several environments: the camera
    raises instead of reporting one environment's image as the batch."""
    env, camera = _make_rasterized_env(batch_size=2)
    with pytest.raises(RuntimeError, match="batched renderer"):
        env.reset(seed=0, reload=True)
    env.close()


# ============================================================================
# NYX Backend Tests
# ============================================================================

def _nyx_cam_opt(**overrides) -> CameraSensorNodeOptions:
    """Create NYX camera options with sensible defaults."""
    kw = dict(
        res=RES,
        pos=CAM_POS,
        quat=CameraSensorNodeOptions.quat_from_lookat(CAM_POS, CAM_LOOKAT),
        fov=45,
        backend="nyx",
        spp=8,  # Low spp for fast tests
    )
    kw.update(overrides)
    return CameraSensorNodeOptions(**kw)


def _make_nyx_unbatched_env(render_depth=False, render_seg=False, render_normal=False, attach=False):
    """Create unbatched environment with NYX camera."""
    world = GenesisWorld()
    box = DummyBoxNode(world)
    
    cam_opts = _nyx_cam_opt(attach=AttachFromName("box") if attach else None)
    
    camera = CameraSensorNode(
        world=world,
        name="cam",
        option=cam_opts,
        render_rgb=True,
        render_depth=render_depth,
        render_seg=render_seg,
        render_normal=render_normal,
    )
    env = WorldEnv(world, [box, camera], render_mode=None)
    return env, camera


@pytest.mark.skipif(not _nyx_available(), reason="NYX not available")
def test_nyx_rgb_only_shapes():
    """NYX backend with RGB only should produce correct shapes."""
    env, camera = _make_nyx_unbatched_env()
    env.reset(seed=0, reload=True)
    cam_obs = camera.get_observation()

    W, H = RES
    assert cam_obs["rgb"].shape == (H, W, 3)
    assert cam_obs["intrinsics_cv"].shape == (3, 3)
    assert cam_obs["extrinsics_cv"].shape == (3, 4)
    assert cam_obs["local_extrinsics_cv"].shape == (3, 4)
    
    # Should not have depth/seg/normal
    assert "depth" not in cam_obs
    assert "seg" not in cam_obs
    assert "normal" not in cam_obs
    env.close()


@pytest.mark.skipif(not _nyx_available(), reason="NYX not available")
def test_nyx_hybrid_shapes():
    """NYX backend with depth should use hybrid rendering (NYX RGB + genesis depth)."""
    env, camera = _make_nyx_unbatched_env(render_depth=True, render_seg=True, render_normal=True)
    env.reset(seed=0, reload=True)
    cam_obs = camera.get_observation()

    W, H = RES
    # RGB from NYX
    assert cam_obs["rgb"].shape == (H, W, 3)
    # Depth/seg/normal from genesis
    assert cam_obs["depth"].shape == (H, W)
    assert cam_obs["seg"].shape == (H, W)
    assert cam_obs["normal"].shape == (H, W, 3)
    # Intrinsics/extrinsics
    assert cam_obs["intrinsics_cv"].shape == (3, 3)
    assert cam_obs["extrinsics_cv"].shape == (3, 4)
    assert cam_obs["local_extrinsics_cv"].shape == (3, 4)
    env.close()


@pytest.mark.skipif(not _nyx_available(), reason="NYX not available")
def test_nyx_intrinsics_values():
    """NYX intrinsics should be computed from FOV."""
    env, camera = _make_nyx_unbatched_env()
    env.reset(seed=0, reload=True)
    K = camera.get_observation()["intrinsics_cv"]

    assert K.shape == (3, 3)
    # fx == fy (single focal length)
    assert K[0, 0] == K[1, 1]
    # Bottom row is [0, 0, 1]
    assert K[2, 0] == 0.0
    assert K[2, 1] == 0.0
    assert K[2, 2] == 1.0
    # Skew is zero
    assert K[0, 1] == 0.0
    assert K[1, 0] == 0.0
    
    # Verify focal length matches FOV
    W, H = RES
    import math
    expected_f = (H / 2.0) / math.tan(math.radians(45) / 2.0)
    assert abs(K[0, 0].item() - expected_f) < 1.0
    env.close()


@pytest.mark.skipif(not _nyx_available(), reason="NYX not available")
def test_nyx_extrinsics_values():
    """NYX extrinsics should be valid rotation matrices."""
    env, camera = _make_nyx_unbatched_env()
    env.reset(seed=0, reload=True)
    E = camera.get_observation()["extrinsics_cv"]

    assert E.shape == (3, 4)
    R = E[:, :3]
    # R should be a valid rotation: R @ R^T ~= I, det(R) ~= 1
    assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-4)
    assert torch.allclose(torch.det(R), torch.tensor(1.0), atol=1e-4)
    env.close()


@pytest.mark.skipif(not _nyx_available(), reason="NYX not available")
def test_nyx_local_extrinsics_static():
    """Static NYX camera: local_extrinsics_cv should equal extrinsics_cv."""
    env, camera = _make_nyx_unbatched_env(attach=False)
    env.reset(seed=0, reload=True)
    cam_obs = camera.get_observation()
    
    # For static camera, local == world extrinsics
    assert torch.allclose(
        cam_obs["extrinsics_cv"],
        cam_obs["local_extrinsics_cv"],
        atol=1e-5
    )
    env.close()


@pytest.mark.skipif(not _nyx_available(), reason="NYX not available")
def test_nyx_local_extrinsics_attached():
    """Attached NYX camera: local_extrinsics_cv should be valid."""
    env, camera = _make_nyx_unbatched_env(attach=True)
    env.reset(seed=0, reload=True)
    cam_obs = camera.get_observation()
    
    local_E = cam_obs["local_extrinsics_cv"]
    world_E = cam_obs["extrinsics_cv"]
    
    # Verify local extrinsics rotation is valid
    R_local = local_E[:, :3]
    assert torch.allclose(R_local @ R_local.T, torch.eye(3), atol=1e-4)
    assert torch.allclose(torch.det(R_local), torch.tensor(1.0), atol=1e-4)
    
    # Verify world extrinsics rotation is valid
    R_world = world_E[:, :3]
    assert torch.allclose(R_world @ R_world.T, torch.eye(3), atol=1e-4)
    assert torch.allclose(torch.det(R_world), torch.tensor(1.0), atol=1e-4)
    
    # For a fixed box at (0,0,0.2), the link's world transform might be identity
    # (with the entity's base position handling the translation), so local and
    # world extrinsics could be similar. Just verify they're both valid.
    assert local_E.shape == (3, 4)
    assert world_E.shape == (3, 4)
    env.close()


@pytest.mark.skipif(not _nyx_available(), reason="NYX not available")
def test_nyx_observation_space_matches():
    """NYX observation space shape should match actual observation shapes."""
    env, camera = _make_nyx_unbatched_env(render_depth=True)
    env.reset(seed=0, reload=True)
    cam_obs = camera.get_observation()

    for key, space in camera.observation_space.spaces.items():
        assert key in cam_obs, f"Key {key!r} in obs space but not in obs"
        assert cam_obs[key].shape == space.shape, (
            f"Shape mismatch for {key!r}: obs={cam_obs[key].shape}, space={space.shape}"
        )
    env.close()


# ============================================================================
# Entity-only attachment normalization (link=None -> entity base link)
# ============================================================================

def test_genesis_entity_only_attach_normalizes_to_base_link():
    """Genesis backend: entity-only attach resolves to the entity's base link."""
    world = GenesisWorld()
    box = DummyBoxNode(world)
    camera = CameraSensorNode(
        world=world,
        name="cam",
        option=_cam_opt(attach=AttachFromName("box")),
        render_rgb=True,
    )
    env = WorldEnv(world, [box, camera], render_mode=None)
    env.reset(seed=0, reload=True)

    entity = world.scene.get_entity("box")
    assert camera._attach_entity is entity
    assert camera._attach_link is entity.base_link
    env.close()


_T_OPENGL_TO_OPENCV = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))


def _camera_to_world_gl(camera) -> torch.Tensor:
    """Recover the OpenGL camera-to-world matrix from reported extrinsics."""
    E = camera.get_observation()["extrinsics_cv"].detach().cpu().float()
    E4 = torch.eye(4)
    E4[:3] = E
    return torch.linalg.inv(E4) @ _T_OPENGL_TO_OPENCV


def _expected_cam_to_world(camera, link) -> torch.Tensor:
    """Expected OpenGL camera-to-world: link world transform @ offset_T."""
    link_pos = torch.as_tensor(link.get_pos(), dtype=torch.float32).cpu()
    link_quat = torch.as_tensor(link.get_quat(), dtype=torch.float32).cpu()
    R = quaternion_to_matrix(PyTorchComputeBackend, link_quat).float()
    link_T = torch.eye(4)
    link_T[:3, :3] = R
    link_T[:3, 3] = link_pos
    offset_T = torch.from_numpy(camera._offset_T).float()
    return link_T @ offset_T


@pytest.mark.skipif(not _nyx_available(), reason="NYX not available")
def test_nyx_entity_only_attach_extrinsics_follow_entity():
    """NYX entity-only attach: world extrinsics track the entity base link.

    The camera is attached with ``link=None``; both rendering and the
    reported ``extrinsics_cv`` must use the entity's base link.  Moving and
    rotating the entity must therefore move the reported camera pose by the
    same amount (a static world pose would be the pre-fix behaviour).
    """
    world = GenesisWorld()
    box = MovableBoxNode(world)
    offset_pos = (0.1, 0.0, 0.05)
    cam_opt = _nyx_cam_opt(
        pos=offset_pos,
        quat=CameraSensorNodeOptions.quat_from_lookat(offset_pos, (0.0, 0.0, 0.0)),
        attach=AttachFromName("box"),
    )
    camera = CameraSensorNode(
        world=world,
        name="cam",
        option=cam_opt,
        render_rgb=True,
    )
    env = WorldEnv(world, [box, camera], render_mode=None)
    env.reset(seed=0, reload=True)

    entity = world.scene.get_entity("box")
    # Entity-only attachment is normalized to the base link.
    assert camera._attach_entity is entity
    assert camera._attach_link is entity.base_link

    # World extrinsics equal the base-link transform composed with offset_T.
    ctw = _camera_to_world_gl(camera)
    expected = _expected_cam_to_world(camera, entity.base_link)
    assert torch.allclose(ctw, expected, atol=1e-4), (
        f"Initial extrinsics do not match base_link @ offset_T:\n{ctw}\nvs\n{expected}"
    )

    cam_pos_before = ctw[:3, 3].clone()
    link_pos_before = torch.as_tensor(
        entity.base_link.get_pos(), dtype=torch.float32,
    ).cpu()

    # Move and rotate the entity; the reported camera pose must follow.
    new_pos = (0.3, -0.2, 0.5)
    yaw = math.radians(90.0)
    entity.set_pos(new_pos)
    entity.set_quat((math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))
    camera.update_observation()

    ctw2 = _camera_to_world_gl(camera)
    expected2 = _expected_cam_to_world(camera, entity.base_link)
    assert torch.allclose(ctw2, expected2, atol=1e-4), (
        f"Extrinsics after moving entity do not match base_link @ offset_T:\n"
        f"{ctw2}\nvs\n{expected2}"
    )

    # The camera actually moved (not a static world pose).  The exact camera
    # pose — including the offset rotated by the entity's new orientation —
    # is already covered by the composition assertions above.
    link_pos_after = torch.as_tensor(
        entity.base_link.get_pos(), dtype=torch.float32,
    ).cpu()
    assert not torch.allclose(link_pos_after, link_pos_before)
    assert not torch.allclose(ctw2[:3, 3], cam_pos_before, atol=1e-3)
    env.close()
