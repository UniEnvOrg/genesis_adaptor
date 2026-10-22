"""ZED camera option helpers: field of view per mode and the near clipping plane."""

import pytest

from unienv_genesis_collection.sensors.cameras.zed import (
    ZED_2I_NARROWLEN_FOVS,
    ZED_2I_WIDELEN_FOVS,
    ZED_MINI_FOVS,
    ZED_NEAR_CLIP,
    ZedSupportedResolution,
    get_zed_2i_narrowlen_camera_options,
    get_zed_2i_widelen_camera_options,
    get_zed_mini_camera_options,
)

_HELPERS = (
    (get_zed_mini_camera_options, ZED_MINI_FOVS),
    (get_zed_2i_widelen_camera_options, ZED_2I_WIDELEN_FOVS),
    (get_zed_2i_narrowlen_camera_options, ZED_2I_NARROWLEN_FOVS),
)


@pytest.mark.parametrize("helper, fovs", _HELPERS)
def test_zed_options_clip_near_the_camera_not_at_the_min_depth(helper, fovs):
    mode = ZedSupportedResolution.HD720
    options = helper(mode)
    assert options.near == ZED_NEAR_CLIP
    assert 0.0 < ZED_NEAR_CLIP <= 0.01
    assert options.fov == fovs[mode]
    assert tuple(options.res) == mode.size


@pytest.mark.parametrize("helper, fovs", _HELPERS)
def test_zed_options_keep_caller_overrides(helper, fovs):
    options = helper(ZedSupportedResolution.HD720, near=0.3, backend="nyx")
    assert options.near == 0.3
    assert options.backend == "nyx"
