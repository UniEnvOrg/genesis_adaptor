from unienv_genesis_collection.objects.mesh_object import MeshObject


class YCBManipulatableObject(MeshObject):
    """A manipulatable object based on a YCB model mesh which is loaded from YCB database

    Args:
        mesh_file: Path to mesh file (``.obj``, ``.stl``, ``.ply``, …).
        model_id: YCB model ID (e.g. "003_cracker_box").
        scale: Uniform scalar scale factor.
        **kwargs: Forwarded to :class:`MeshObject"""

    def __init__(
        self,
        mesh_file,
        model_id: str,
        scale: float = 1.0,
        annotations=None,
        **kwargs,
    ):
        super().__init__(mesh_file=mesh_file, scale=scale, **kwargs)
        self.model_id = model_id
        if annotations is not None:
            self.annotations = annotations