"""Live view of the hand's taxel readings, painted on the flat palmar schematic."""
import math

import cv2
import numpy as np

from bake import face_row
from common import SIDE, link_mesh, out_path

# The hand reads gray so the taxels carry all the colour. RGB throughout;
# ``show`` is the only place that flips to the BGR cv2 wants.
BG = (238, 239, 241)
BODY = (152, 156, 160)
EDGE = (86, 90, 94)
COLD = np.array([255.0, 255.0, 255.0])   # no contact
HOT = np.array([214.0, 32.0, 28.0])      # vmax or more

PAD = 2.2        # mm of overlap so segment seams have no gap
INSET = 0.16     # fraction of each pad given up to the gray body around it
BAND = 0.78      # of the drawn link's length; keeps segments from painting over
                 # each other, since their projected boxes overlap by about a row
REACH = 52       # mm the connector runs back toward the palm
MARGIN_PX = 8

VMAX_DEPTH = 0.008   # m of indentation that reads as full red
VMAX_FORCE = 2.0     # N of taxel force that reads as full red


def _even(n):
    """Round up to an even number of pixels; h264 rejects odd frame sides."""
    return int(n) + int(n) % 2


def _frame(row, pad=PAD):
    """Face centre, along-axis unit vector, across unit vector, and padded size."""
    a = math.radians(row["ang"])
    u = np.array([math.cos(a), math.sin(a)])     # along the link's long axis
    v = np.array([-u[1], u[0]])                  # across it (the wrapped arc)
    return np.array(row["c"], float), u, v, row["h"] + pad * 2, row["w"] + pad * 2


def _quad(c, u, v, H, W, su0, su1, sv0, sv1):
    return np.array([c + u * H * su + v * W * sv
                     for su, sv in ((su0, sv0), (su0, sv1), (su1, sv1), (su1, sv0))])


def _rect(row):
    return _quad(*_frame(row), -.5, .5, -.5, .5)


def _tiles(row, n_around, n_along, inset=INSET, band=BAND):
    """The face cut into ``n_along`` rows along the link by ``n_around`` columns across it."""
    # Row-major in the same order _arc_probes places its probes, so tile k is
    # taxel k; the columns run the other way round the band than v, hence the
    # reversed j.
    c, u, v, H, W = _frame(row)
    H *= band
    for i in range(n_along):
        for j in range(n_around):
            col = n_around - 1 - j
            tile = _quad(
                c, u, v, H, W,
                i / n_along - .5, (i + 1) / n_along - .5,
                col / n_around - .5, (col + 1) / n_around - .5,
            )
            # shrink each pad so the gray link shows through as its border
            yield i * n_around + j, tile.mean(0) + (tile - tile.mean(0)) * (1 - inset)


def _connector(quad, palm_c):
    """A gray strip bridging link1, which has no collision geometry and so no taxels."""
    order = sorted(range(4), key=lambda i: np.linalg.norm(quad[i] - palm_c))
    a, b = quad[order[0]], quad[order[1]]
    dv = palm_c - (a + b) / 2
    dv /= max(np.linalg.norm(dv), 1e-9)
    return np.array([a, b, b + dv * REACH, a + dv * REACH])


def bake_layout(node):
    """The per-link face table, measured from the links the node is attached to."""
    links = {}
    for attach in node.option.attaches:
        entity, link = attach.get_attachment(node.world)
        links[attach.link_name] = link if link is not None else entity.base_link
    palm = next(lk for name, lk in links.items() if name.endswith("wuji_palm"))
    return [face_row(name, lk, link_mesh(lk), palm) for name, lk in links.items()]


class TactileView:
    """Draws a TactileSensorNode's readings on the open palm, and records them."""

    def __init__(self, node, *, vmax=None, width=560, layout=None):
        if width < 64:
            raise ValueError(f"width must be at least 64 px, got {width}")
        self.node = node
        self.kinematic = node.option.sensor_type == "kinematic"
        self.vmax = float(
            vmax if vmax is not None else (VMAX_FORCE if self.kinematic else VMAX_DEPTH)
        )
        if self.vmax <= 0.0:
            raise ValueError(f"vmax must be positive, got {self.vmax}")
        self.n_around = node.option.n_around
        self.n_along = node.option.n_along
        self.n_taxels = node.n_taxels
        self.rows = list(layout) if layout is not None else bake_layout(node)
        self._place(width)
        self._frames = []
        self._window = None

    # ------------------------------------------------------------- layout ---
    def _place(self, width):
        """Turn the mm face table into the pixel polygons drawn every frame."""
        palm_c = np.array(
            next(r for r in self.rows if r["link"].endswith("wuji_palm"))["c"], float
        )
        rects = [(_rect(r), r) for r in self.rows]
        conns = [_connector(q, palm_c) for q, r in rects if r["link"].endswith("link2")]

        pts = np.concatenate([q for q, _ in rects] + conns)
        lo, hi = pts.min(axis=0) - 6, pts.max(axis=0) + 6
        scale = (width - 2 * MARGIN_PX) / (hi[0] - lo[0])
        # h264 wants even side lengths, so round both up to even
        self.width = _even(width)
        self.height = _even(int(round((hi[1] - lo[1]) * scale)) + 2 * MARGIN_PX)

        def px(quad):
            # mirror in x for handedness and flip y so the fingers point up,
            # the same mapping the static schematic uses
            out = np.stack([(hi[0] - quad[:, 0]), (hi[1] - quad[:, 1])], axis=1)
            return np.round(out * scale + MARGIN_PX).astype(np.int32)

        self._body = [px(c) for c in conns]
        self._outlines = [px(q) for q, _ in rects]
        self._taxels = [
            (r["link"], k, px(tile))
            for _, r in rects
            for k, tile in _tiles(r, self.n_around, self.n_along)
        ]

    # -------------------------------------------------------------- values ---
    def _readings(self, obs, env):
        """Per-link taxel magnitudes as ``(N,)`` numpy, one env's worth."""
        if self.node.name in obs:
            obs = obs[self.node.name]
        missing = [r["link"] for r in self.rows if r["link"] not in obs]
        if missing:
            raise ValueError(
                f"observation is missing taxel readings for {missing}; it has "
                f"{sorted(obs)}"
            )
        out = {}
        for row in self.rows:
            value = obs[row["link"]]
            if self.kinematic:
                value = value["force"]
            value = np.asarray(
                value.detach().cpu() if hasattr(value, "detach") else value, float
            )
            if self.kinematic:
                value = np.linalg.norm(value, axis=-1)
            if value.ndim > 1:
                value = value[env]
            if value.shape != (self.n_taxels,):
                raise ValueError(
                    f"link '{row['link']}' gave {value.shape} readings, expected "
                    f"({self.n_taxels},)"
                )
            out[row["link"]] = value
        return out

    def _colour(self, value):
        t = float(np.clip(value / self.vmax, 0.0, 1.0))
        return tuple(int(round(x)) for x in COLD + (HOT - COLD) * t)

    # -------------------------------------------------------------- render ---
    def update(self, obs, env=0):
        """Return the RGB ``(H, W, 3)`` picture of this observation's taxels."""
        readings = self._readings(obs, env)
        img = np.full((self.height, self.width, 3), BG, np.uint8)
        for poly in self._body + self._outlines:
            cv2.fillConvexPoly(img, poly, BODY, cv2.LINE_AA)
        for link, k, poly in self._taxels:
            cv2.fillConvexPoly(img, poly, self._colour(readings[link][k]), cv2.LINE_AA)
        for poly in self._outlines:
            cv2.polylines(img, [poly], True, EDGE, 2, cv2.LINE_AA)
        return img

    def show(self, obs, env=0, title="hand tactile"):
        """Draw this observation in a popup window and return the picture."""
        img = self.update(obs, env)
        self._window = title
        cv2.imshow(title, img[:, :, ::-1])
        cv2.waitKey(1)
        return img

    def record(self, obs, scene_rgb=None, env=0):
        """Keep this observation's picture for ``save``, with the scene beside it."""
        img = self.update(obs, env)
        if scene_rgb is not None:
            img = np.hstack([img, _fit_height(scene_rgb, img.shape[0], env)])
        self._frames.append(img)
        return img

    def save(self, path=None, fps=15):
        """Write the recorded frames as one mp4 and return where it went."""
        import imageio

        if not self._frames:
            raise RuntimeError("nothing recorded yet; call record() first")
        path = path or out_path("tactile.mp4")
        with imageio.get_writer(path, fps=fps, macro_block_size=1) as writer:
            for frame in self._frames:
                writer.append_data(frame)
        print(f"wrote {path}  ({len(self._frames)} frames, "
              f"{self._frames[0].shape[1]}x{self._frames[0].shape[0]})")
        return path

    def close(self):
        if self._window is not None:
            cv2.destroyWindow(self._window)
            self._window = None


def _fit_height(rgb, height, env):
    """A camera's rgb observation as uint8, scaled to sit beside the hand."""
    img = np.asarray(rgb.detach().cpu() if hasattr(rgb, "detach") else rgb)
    if img.ndim == 4:
        img = img[env]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    width = _even(max(int(round(img.shape[1] * height / img.shape[0])), 2))
    return cv2.resize(img[..., :3], (width, height), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------- demo ---
def build_hand_with_tactile():
    """A built scene holding the Tianji robot and one hand's taxels; returns ``(env, node)``."""
    # gs.init() is the caller's responsibility throughout this repo; a script is
    # the caller, so it happens here rather than in library code.
    import genesis as gs

    if not gs._initialized:
        from genesis.constants import backend as gs_backend

        gs.init(backend=gs_backend.cpu)

    from unienv_interface.world import WorldEnv
    from unienv_genesis import GenesisWorld
    from unienv_genesis_collection.robots.tianji_marvin_wuji import TianjiMarvinWuji
    from unienv_genesis_collection.robots.wuji_tactile import make_wuji_hand_tactile

    world = GenesisWorld()
    robot = TianjiMarvinWuji(world, name="tianji")
    node = make_wuji_hand_tactile(world, robot_name="tianji", sides=(SIDE,))
    env = WorldEnv(world, [robot, node], render_mode=None)
    env.reset(seed=0, reload=True)
    return env, node


def _sweep(view, step, steps):
    """A band of contact travelling from the palm out to the fingertips."""
    obs = {}
    for i, row in enumerate(view.rows):
        phase = (i + np.arange(view.n_taxels) / view.n_taxels) / len(view.rows)
        wave = np.cos(2 * np.pi * (step / steps - phase))
        obs[row["link"]] = view.vmax * np.clip(wave, 0.0, 1.0) ** 3
    return obs


if __name__ == "__main__":
    env, node = build_hand_with_tactile()
    view = TactileView(node)
    env.close()

    steps = 60
    for step in range(steps):
        view.record(_sweep(view, step, steps))
    view.save(out_path("tactile_demo.mp4"))
