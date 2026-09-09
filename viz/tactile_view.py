import cv2
import numpy as np

from artifacts import compute_face_row
from common import build_hand, get_link_mesh, get_output_path
from layout import compute_bounds, compute_connector, compute_rectangle, generate_tiles, transform_to_canvas

BACKGROUND_COLOR = (238, 239, 241)
BODY_COLOR = (152, 156, 160)
OUTLINE_COLOR = (86, 90, 94)
COLD_COLOR = np.array([255.0, 255.0, 255.0])
HOT_COLOR = np.array([214.0, 32.0, 28.0])
TAXEL_INSET = 0.16
TAXEL_BAND_LENGTH = 0.78
IMAGE_MARGIN = 8
MAX_DEPTH = 0.008
MAX_FORCE = 2.0


def _round_up_even(value):
    return int(value) + int(value) % 2


def compute_open_hand_layout(node):
    links = {}
    entity = None
    for attachment in node.option.attaches:
        attached_entity, link = attachment.get_attachment(node.world)
        entity = attached_entity if entity is None else entity
        if attached_entity is not entity:
            raise ValueError("TactileView requires every attachment to belong to one entity")
        links[attachment.link_name] = link if link is not None else attached_entity.base_link

    palm_name = next(name for name in links if name.endswith("wuji_palm"))
    side = palm_name.removesuffix("_wuji_palm")
    open_qpos = entity.get_qpos().detach().clone()
    for joint in entity.joints:
        if joint.name.startswith(f"{side}_finger"):
            open_qpos[..., joint.qs_idx_local] = 0.0

    names = list(links)
    positions, quaternions = entity.forward_kinematics(
        open_qpos, links_idx_local=[links[name].idx_local for name in names]
    )
    if positions.ndim == 3:
        positions, quaternions = positions[0], quaternions[0]
    poses = {
        name: (position.detach().cpu().numpy(), quaternion.detach().cpu().numpy())
        for name, position, quaternion in zip(names, positions, quaternions)
    }
    palm_pose = poses[palm_name]
    return [compute_face_row(name, get_link_mesh(links[name]), poses[name], palm_pose) for name in names]


class TactileView:
    def __init__(self, node, *, vmax=None, width=560, layout=None):
        if width < 64:
            raise ValueError(f"width must be at least 64 px, got {width}")
        self.node = node
        self.kinematic = node.option.sensor_type == "kinematic"
        self.vmax = float(vmax if vmax is not None else (MAX_FORCE if self.kinematic else MAX_DEPTH))
        if self.vmax <= 0.0:
            raise ValueError(f"vmax must be positive, got {self.vmax}")
        self.n_around = node.option.n_around
        self.n_along = node.option.n_along
        self.n_taxels = node.n_taxels
        self.rows = list(layout) if layout is not None else compute_open_hand_layout(node)
        self._place_layout(width)
        self._frames = []
        self._window = None

    def _place_layout(self, width):
        palm = next(row for row in self.rows if row["link"].endswith("wuji_palm"))
        rectangles = [(compute_rectangle(row), row) for row in self.rows]
        connectors = [
            compute_connector(rectangle, palm["c"])
            for rectangle, row in rectangles
            if row["link"].endswith("link2")
        ]
        lower, upper = compute_bounds([rectangle for rectangle, _ in rectangles] + connectors)
        scale = (width - 2 * IMAGE_MARGIN) / (upper[0] - lower[0])
        self.width = _round_up_even(width)
        self.height = _round_up_even(int(round((upper[1] - lower[1]) * scale)) + 2 * IMAGE_MARGIN)

        def to_pixels(points):
            return np.round(transform_to_canvas(points, upper, scale, IMAGE_MARGIN)).astype(np.int32)

        self._body = [to_pixels(connector) for connector in connectors]
        self._outlines = [to_pixels(rectangle) for rectangle, _ in rectangles]
        self._taxels = [
            (row["link"], index, to_pixels(tile))
            for _, row in rectangles
            for index, tile in generate_tiles(
                row,
                self.n_around,
                self.n_along,
                inset=TAXEL_INSET,
                length_scale=TAXEL_BAND_LENGTH,
                reverse_across=True,
            )
        ]

    def _get_readings(self, observation, environment_index):
        if self.node.name in observation:
            observation = observation[self.node.name]
        missing = [row["link"] for row in self.rows if row["link"] not in observation]
        if missing:
            raise ValueError(f"observation is missing taxel readings for {missing}; it has {sorted(observation)}")

        readings = {}
        for row in self.rows:
            value = observation[row["link"]]
            if self.kinematic:
                value = value["force"]
            value = np.asarray(value.detach().cpu() if hasattr(value, "detach") else value, float)
            if self.kinematic:
                value = np.linalg.norm(value, axis=-1)
            if value.ndim > 1:
                value = value[environment_index]
            if value.shape != (self.n_taxels,):
                raise ValueError(f"link '{row['link']}' gave {value.shape} readings, expected ({self.n_taxels},)")
            readings[row["link"]] = value
        return readings

    def _compute_colour(self, value):
        intensity = float(np.clip(value / self.vmax, 0.0, 1.0))
        return tuple(int(round(channel)) for channel in COLD_COLOR + (HOT_COLOR - COLD_COLOR) * intensity)

    def update(self, obs, env=0):
        readings = self._get_readings(obs, env)
        image = np.full((self.height, self.width, 3), BACKGROUND_COLOR, np.uint8)
        for polygon in self._body + self._outlines:
            cv2.fillConvexPoly(image, polygon, BODY_COLOR, cv2.LINE_AA)
        for link, index, polygon in self._taxels:
            cv2.fillConvexPoly(image, polygon, self._compute_colour(readings[link][index]), cv2.LINE_AA)
        for polygon in self._outlines:
            cv2.polylines(image, [polygon], True, OUTLINE_COLOR, 2, cv2.LINE_AA)
        return image

    def show(self, obs, env=0, title="hand tactile"):
        image = self.update(obs, env)
        self._window = title
        cv2.imshow(title, image[:, :, ::-1])
        cv2.waitKey(1)
        return image

    def record(self, obs, scene_rgb=None, env=0):
        image = self.update(obs, env)
        if scene_rgb is not None:
            image = np.hstack([image, _resize_to_height(scene_rgb, image.shape[0], env)])
        self._frames.append(image)
        return image

    def save(self, path=None, fps=15):
        import imageio

        if not self._frames:
            raise RuntimeError("nothing recorded yet; call record() first")
        path = path or get_output_path("tactile.mp4")
        with imageio.get_writer(path, fps=fps, macro_block_size=1) as writer:
            for frame in self._frames:
                writer.append_data(frame)
        first = self._frames[0]
        print(f"wrote {path}  ({len(self._frames)} frames, {first.shape[1]}x{first.shape[0]})")
        return path

    def close(self):
        if self._window is not None:
            cv2.destroyWindow(self._window)
            self._window = None


def _resize_to_height(rgb, height, environment_index):
    image = np.asarray(rgb.detach().cpu() if hasattr(rgb, "detach") else rgb)
    if image.ndim == 4:
        image = image[environment_index]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    width = _round_up_even(max(int(round(image.shape[1] * height / image.shape[0])), 2))
    return cv2.resize(image[..., :3], (width, height), interpolation=cv2.INTER_AREA)


def build_hand_with_tactile():
    return build_hand(with_tactile=True)


def _generate_sweep(view, step, steps):
    observation = {}
    for index, row in enumerate(view.rows):
        phase = (index + np.arange(view.n_taxels) / view.n_taxels) / len(view.rows)
        wave = np.cos(2 * np.pi * (step / steps - phase))
        observation[row["link"]] = view.vmax * np.clip(wave, 0.0, 1.0) ** 3
    return observation


if __name__ == "__main__":
    environment, node = build_hand_with_tactile()
    view = TactileView(node)
    environment.close()
    for step in range(60):
        view.record(_generate_sweep(view, step, 60))
    view.save(get_output_path("tactile_demo.mp4"))
