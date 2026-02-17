"""Asset path utilities for unienv_genesis_collection."""

import os
from pathlib import Path
from typing import Union

__all__ = [
    "ASSETS_DIR",
    "get_asset_path",
]

# Base path to the assets directory
ASSETS_DIR = Path(__file__).parent.resolve() / "assets"

def get_asset_path(*relative_paths: Union[str, os.PathLike]) -> Path:
    """Get the absolute path to an asset file.
    
    This function resolves a relative path within the assets directory
    to an absolute Path.
    
    Args:
        *relative_paths: Path components relative to the assets directory.
            Can be a single string like "robots/franka_fr3/fr3.xml" or
            multiple components like ("robots", "franka_fr3", "fr3.xml")
    
    Returns:
        Absolute Path to the asset file
    
    Raises:
        FileNotFoundError: If the asset file does not exist
    
    Example:
        >>> from unienv_genesis_collection.asset_path import get_asset_path
        >>> # Single string argument
        >>> fr3_path = get_asset_path("robots/franka_fr3/fr3.xml")
        >>> # Multiple path components
        >>> fr3_path = get_asset_path("robots", "franka_fr3", "fr3.xml")
        >>> print(fr3_path)
        Path('/path/to/unienv_genesis_collection/assets/robots/franka_fr3/fr3.xml')
    """
    # Join all path components
    if len(relative_paths) == 1 and isinstance(relative_paths[0], (str, os.PathLike)):
        # Single argument case - could be a string path with separators
        relative_path = relative_paths[0]
    else:
        # Multiple arguments - join them
        relative_path = os.path.join(*relative_paths)
    
    absolute_path = ASSETS_DIR / relative_path
    
    if not absolute_path.exists():
        raise FileNotFoundError(f"Asset not found: {relative_path} (looked in {absolute_path})")
    
    return absolute_path

