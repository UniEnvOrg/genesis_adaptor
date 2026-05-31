from typing import Literal, Optional, Any, Sequence, Dict
from dataclasses import dataclass, field
import os

# Right now Genesis has not-unified options for adding lights to the scene, so we'll declare them here and forward to the appropriate place depending on the renderer used.
@dataclass(frozen=True)
class GenesisWorldLightOption:
    light_type : Literal["directional", "point", "spot"] = "directional"
    """Type of light to add. One of "directional", "point", or "spot"."""
    pos : tuple[float, float, float] | None = None
    """Position of the light. For directional lights, this is ignored."""
    direction : tuple[float, float, float] | None = None
    """Direction of the light. For point lights, this is ignored."""
    color : tuple[float, float, float] = (1.0, 1.0, 1.0)
    """RGB color of the light, each component in [0, 1]."""
    intensity : float = 1.0
    """Intensity multiplier for the light."""
    range : float = 3.0
    """Range of the light. Only applicable for point and spot lights."""
    spot_inner_cone_angle : Optional[float] = None
    """Inner cone angle for spot lights, in degrees."""
    spot_outer_cone_angle : Optional[float] = None
    """Outer cone angle for spot lights, in degrees."""
    attenuation : float = 0.0
    """The attenuation factor of the light.
    Light intensity will attenuate by distance with (1 / (1 + attenuation * distance ^ 2)). This is only supported for BatchRenderer.
    """
    cast_shadow : bool = False
    """Whether the light casts shadows. Only supported for directional and spot lights in BatchRenderer."""

    def to_light_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary format expected by the renderer."""
        light_dict = {
            "type": self.light_type,
            "color": self.color,
            "intensity": self.intensity,
        }
        if self.light_type == "directional":
            light_dict["dir"] = self.direction if self.direction is not None else (0.0, 0.0, -1.0)
        elif self.light_type in ["point", "spot"]:
            light_dict["pos"] = self.pos if self.pos is not None else (0.0, 0.0, 1.0)
            light_dict["range"] = self.range
            if self.light_type == "spot":
                light_dict["inner_angle"] = self.spot_inner_cone_angle
                light_dict["outer_angle"] = self.spot_outer_cone_angle
        return light_dict

@dataclass(frozen=True)
class GenesisWorldEnvMapOption:    
    hdr_path : str | os.PathLike | None = None
    """File path to the HDR environment map to use for lighting. Only applicable for nyx camera. If this is None, the nyx renderer will use the tint color as a solid environment."""

    png_path : str | os.PathLike | None = None
    """File path to a non-HDR image to use as the environment map. Only applicable for ray tracer camera."""

    hdr_layout : Literal['LongLat', 'Column', 'Row', 'HorizontalCross', 'VerticalCross'] = 'LongLat'
    """Layout of the environment map."""

    hdr_multiplier : float = 1.0
    """Scalar multiplier applied to the environment radiance. Default is 1.0 (no change)."""

    hdr_tint: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """RGB tint applied to the environment map. Each component is in [0, 1], and default is (1.0, 1.0, 1.0) for no tint."""

    hdr_rotation: float = 0.0
    """Rotation (in degrees) around the world up axis to apply to the environment map. Default is 0.0 (no rotation)."""

    png_pos : tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position of the environment map when using a non-HDR image. Default is (0.0, 0.0, 0.0)."""

    png_euler: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Rotation (in degrees) to apply to the environment map when using a non-HDR image. Default is 0.0 (no rotation)."""

    png_radius : float = 15.0

    def to_hyx_asset(self):
        
        import gs_nyx.nyx_py_sdk as nps
        env_map = nps.EnvironmentMapAsset()
        env_map.texture = self.hdr_path
        env_map.layout = getattr(nps.EEnvMapLayout, self.hdr_layout)
        env_map.multiplier = self.hdr_multiplier
        env_map.tint = nps.float3(*self.hdr_tint)
        env_map.rotation = self.hdr_rotation
        return env_map