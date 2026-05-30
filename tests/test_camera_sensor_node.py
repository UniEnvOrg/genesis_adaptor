"""Tests for CameraSensorNode observation shapes and intrinsics/extrinsics (unbatched, CPU)."""

from __future__ import annotations

import pytest
import torch
import genesis as gs
from genesis.constants import backend as gs_backend

if not gs._initialized:
    gs.init(backend=gs_backend.cpu)

from unienv_interface.world import WorldEnv
from unienv_interface.world.node import WorldNode

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.sensors import (
    CameraSensorNode,
    CameraSensorNodeOptions,
)
from unienv_genesis.options.attach_option import AttachFromName

# Check if NYX is available
try:
    import gs_nyx.nyx_py_renderer as npr
    import gs_nyx.nyx_py_sdk as nps
    from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions
    NYX_AVAILABLE = True
except ImportError:
    NYX_AVAILABLE = False


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


@pytest.mark.skipif(not NYX_AVAILABLE, reason="NYX not available")
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


@pytest.mark.skipif(not NYX_AVAILABLE, reason="NYX not available")
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


@pytest.mark.skipif(not NYX_AVAILABLE, reason="NYX not available")
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


@pytest.mark.skipif(not NYX_AVAILABLE, reason="NYX not available")
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


@pytest.mark.skipif(not NYX_AVAILABLE, reason="NYX not available")
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


@pytest.mark.skipif(not NYX_AVAILABLE, reason="NYX not available")
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


@pytest.mark.skipif(not NYX_AVAILABLE, reason="NYX not available")
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
