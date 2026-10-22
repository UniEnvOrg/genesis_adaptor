"""
Common ZED camera parameter definitions
Parameters are taken from: https://support.stereolabs.com/hc/en-us/articles/360007395634-What-is-the-camera-focal-length-and-field-of-view
"""

from enum import Enum
from functools import partial
from unienv_genesis.nodes.sensors.camera_node import CameraSensorNodeOptions

__all__ = [
    "ZedSupportedResolution",
    "ZED_NEAR_CLIP",
    "ZED_MINI_MIN_DEPTH",
    "ZED_2I_WIDELEN_MIN_DEPTH",
    "ZED_2I_NARROWLEN_MIN_DEPTH",
    "get_zed_mini_camera_options",
    "get_zed_2i_widelen_camera_options",
    "get_zed_2i_narrowlen_camera_options",
]

class ZedSupportedResolution(Enum):
    HD4K = (3840, 2160)
    HD2K = (2208, 1242)
    HD1200 = (1920, 1200)
    HD1080 = (1920, 1080)
    HD720 = (1280, 720)
    SVGA = (960, 600)
    WVGA = (672, 376)

    @property
    def width(self) -> int:
        return self.value[0]

    @property
    def height(self) -> int:
        return self.value[1]

    @property
    def size(self) -> tuple[int, int]:
        return self.value

#: Near clipping plane of every ZED camera (m). It is a rendering parameter, not
#: the sensors' minimum depth (:data:`ZED_MINI_MIN_DEPTH` & co.): clipping there
#: would hide close-by objects real RGB images show — the gripper fingers under a
#: wrist camera, a robot's arms under a head-mounted camera. Genesis's rasterizer
#: returns NaN depth at ``near=0`` and NYX requires ``near > 0``, hence 1 cm.
ZED_NEAR_CLIP = 0.01

#: Minimum depth range (m) of each ZED model, for masking simulated depth like the sensor's.
ZED_MINI_MIN_DEPTH = 0.1
ZED_2I_WIDELEN_MIN_DEPTH = 0.3
ZED_2I_NARROWLEN_MIN_DEPTH = 1.5

# Zed Mini camera parameters
ZED_MINI_FOVS = {
    ZedSupportedResolution.HD2K: 46,
    ZedSupportedResolution.HD1080: 40,
    ZedSupportedResolution.HD720: 52,
    ZedSupportedResolution.WVGA: 54
}
ZED_MINI_DEFAULT_PARAM = {
    "near": ZED_NEAR_CLIP,
    "far": 20.0,
    "aperture": 2.0,
}
def get_zed_mini_camera_options(
    res : ZedSupportedResolution,
    **kwargs
) -> CameraSensorNodeOptions:
    assert res in ZED_MINI_FOVS, f"Unsupported resolution for zed mini, list of supported resolutions: {list(ZED_MINI_FOVS.keys())}"
    fov = ZED_MINI_FOVS[res]
    prefill_kwargs = ZED_MINI_DEFAULT_PARAM.copy()
    prefill_kwargs['fov'] = fov
    prefill_kwargs['res'] = res.size
    prefill_kwargs.update(kwargs)
    return CameraSensorNodeOptions(
        **prefill_kwargs
    )

# 2.1mm len zed 2i parameters
ZED_2I_WIDELEN_FOVS = {
    ZedSupportedResolution.HD2K: 61,
    ZedSupportedResolution.HD1080: 54,
    ZedSupportedResolution.HD720: 68,
    ZedSupportedResolution.WVGA: 71
}
ZED_2I_WIDELEN_DEFAULT_PARAM = {
    "near": ZED_NEAR_CLIP,
    "far": 20.0,
    "aperture": 2.0,
}
def get_zed_2i_widelen_camera_options(
    res : ZedSupportedResolution,
    **kwargs
) -> CameraSensorNodeOptions:
    assert res in ZED_2I_WIDELEN_FOVS, f"Unsupported resolution for zed 2i wide (2.1mm) len, list of supported resolutions: {list(ZED_2I_WIDELEN_FOVS.keys())}"
    fov = ZED_2I_WIDELEN_FOVS[res]
    prefill_kwargs = ZED_2I_WIDELEN_DEFAULT_PARAM.copy()
    prefill_kwargs['fov'] = fov
    prefill_kwargs['res'] = res.size
    prefill_kwargs.update(kwargs)
    return CameraSensorNodeOptions(
        **prefill_kwargs
    )

# 4mm len zed 2i parameters
ZED_2I_NARROWLEN_FOVS = {
    ZedSupportedResolution.HD2K: 36,
    ZedSupportedResolution.HD1080: 31,
    ZedSupportedResolution.HD720: 41,
    ZedSupportedResolution.WVGA: 43
}
ZED_2I_NARROWLEN_DEFAULT_PARAM = {
    "near": ZED_NEAR_CLIP,
    "far": 35.0,
    "aperture": 1.8
}
def get_zed_2i_narrowlen_camera_options(
    res : ZedSupportedResolution,
    **kwargs
) -> CameraSensorNodeOptions:
    assert res in ZED_2I_NARROWLEN_FOVS, f"Unsupported resolution for zed 2i narrow (4mm) len, list of supported resolutions: {list(ZED_2I_NARROWLEN_FOVS.keys())}"
    fov = ZED_2I_NARROWLEN_FOVS[res]
    prefill_kwargs = ZED_2I_NARROWLEN_DEFAULT_PARAM.copy()
    prefill_kwargs['fov'] = fov
    prefill_kwargs['res'] = res.size
    prefill_kwargs.update(kwargs)
    return CameraSensorNodeOptions(
        **prefill_kwargs
    )
