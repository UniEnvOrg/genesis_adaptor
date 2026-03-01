from pathlib import Path
import json
import random

from typing import Iterable, Optional
from mani_skill import ASSET_DIR
from unienv_genesis_collection.objects.pick_place_objs import YCBManipulatableObject


YCB_DATASET = dict()

def _load_ycb_dataset():
    global YCB_DATASET
    file_path = ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json"
    raw_file = json.loads(file_path.read_text())
    if isinstance(raw_file, dict) and "model_data" in raw_file and isinstance(raw_file["model_data"], dict):
        model_data = raw_file["model_data"]
    else:
        model_data = raw_file

    YCB_DATASET = {
        "model_data": model_data,
    }

class YCBParser:
    """Parses YCB metadata and creates YCBManipulatableObjects from stored object mesh.

    Expected directory layout under ``root``:

        root/
          info_pick_v0.json
          models/
            <model_id>/
              collision.ply
              textured.obj
    """

    def __init__(
        self,
        allowed_model_ids: Optional[Iterable[str]] = None,
        annotation_filepath: Optional[Path] = None,
    ):
        if "model_data" not in YCB_DATASET:
            _load_ycb_dataset()
        ycb_meshes = YCB_DATASET["model_data"]

        self.root = ASSET_DIR / "assets/mani_skill2_ycb"

        # reading in YCB object annotations from ycb.json
        default_annotation_path = Path(__file__).with_name("ycb.json")
        self.annotation_path = (
            Path(annotation_filepath)
            if annotation_filepath is not None
            else default_annotation_path
        )
        # can replace ycb.json with some other file
        if self.annotation_path.exists():
            annotations = json.loads(self.annotation_path.read_text())
            if not isinstance(annotations, dict):
                raise ValueError(
                    f"Custom YCB metadata at {self.annotation_path} must be a JSON object"
                )
            self.annotations = annotations
        else:
            self.annotations = {}

        effective_allowed_model_ids = allowed_model_ids
        # if no explicit list of working object assets is provided, treat all objects in 
        # YCB annotated file as working objects
        if effective_allowed_model_ids is None and len(self.annotations) > 0:
            effective_allowed_model_ids = self.annotations.keys()

        # if no annotated YCB file either, all Genesis assets are considered working objects
        if effective_allowed_model_ids is None:
            self.ycb_meshes = ycb_meshes
        else:
            allowed = set(effective_allowed_model_ids)
            # ensure all specified working object assets actually exist in Genesis assets
            missing = sorted([model_id for model_id in allowed if model_id not in ycb_meshes])
            if missing:
                raise KeyError(
                    f"Unrecognized YCB model IDs found in allowed list: {missing}"
                )
            # map mesh to model id for working objects
            self.ycb_meshes = {
                model_id: ycb_meshes[model_id]
                for model_id in allowed
                if model_id in ycb_meshes
            }

    @property
    def supported_model_ids(self):
        return sorted(self.ycb_meshes.keys())

    def get_mesh_from_id(self, model_id: str):
        if model_id not in self.ycb_meshes:
            available = sorted(self.ycb_meshes.keys())
            raise KeyError(
                f"Unknown YCB model_id '{model_id}'. Available ids: {available}"
            )
        return self.ycb_meshes[model_id]

    def resolve_mesh_path(self, model_id: str, use_collision: bool = False) -> Path:
        mesh_name = "collision.ply" if use_collision else "textured.obj"
        return self.root / "models" / model_id / mesh_name

    def create(self, model_id: str, scale: float = 1.0, use_collision: bool = False, **kwargs):
        mesh = self.get_mesh_from_id(model_id)
        annotations = self.annotations.get(model_id, {})
        annotated_model = {**mesh, **annotations}

        model_size_low = annotated_model.get("size_low", 1.0)
        model_size_high = annotated_model.get("size_high", 1.0)
        model_scale = random.uniform(model_size_low, model_size_high)
        s = scale * model_scale
        mesh_path = self.resolve_mesh_path(model_id, use_collision=use_collision)

        # we're not actually doing anything with the non-scale annotations in ycb.json
        return YCBManipulatableObject(
            mesh_file=mesh_path,
            model_id=model_id,
            scale=s,
            # annotations=annotated_model,
            name=f"ycb_{model_id}",
            **kwargs,
        )