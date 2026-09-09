# viz — Wuji hand tactile layout

Scripts that draw, render, and measure where `TactileSensorNode` puts its taxels
on the Tianji/Wuji hand. Development tools, not part of the package: they are
excluded from the wheel by the package whitelist in `pyproject.toml`.

Everything reads geometry only — no physics is stepped — so these run on CPU
with no GPU required.

## Running

These need the `genesis` conda env — it has genesis, numpy, PIL, trimesh, and the
`unienv_*` packages. One command builds the artifact page. Paths below are
relative to the repo root; there is no need to `cd` into `viz/` first, since a
script's own directory is always on `sys.path`:

```bash
conda activate genesis
python viz/build.py        # -> out/tactile.html, ~7 s
```

Without activating, point at the env's interpreter directly:

```bash
/home/pranavnnt/libraries/miniconda3/envs/genesis/bin/python viz/build.py
```

It builds the hand once and does the rest in memory. The stages are also
runnable on their own when you only want an intermediate:

```bash
python viz/bake.py         # out/taxels.json + out/hand_{palm,three_quarter,edge}.jpg
python viz/draw_layout.py  # out/taxels.json -> out/hand_layout.svg
```

`draw_layout.py` reads the existing `taxels.json` and builds no scene, so it is
the fast one to iterate on while tweaking the flat schematic.

`tactile_view.py` run on its own makes a demo video from a synthetic contact wave:

```bash
python viz/tactile_view.py   # -> out/tactile_demo.mp4
```

Two diagnostics stand alone:

```bash
python viz/check_probes.py   # on-surface distance, unit outward normals, band width
python viz/measure_wrap.py   # how far around each link the sensed band really reaches
```

Outputs all land in `out/`, which is gitignored.

## Watching the taxels while an env runs

`TactileView` paints one `TactileSensorNode`'s readings on a fixed open-palm layout.
It computes that layout with forward kinematics without changing the environment;
only the pad colours change after construction.

```python
import sys; sys.path.insert(0, "viz")
from tactile_view import TactileView

view = TactileView(node)
_, obs, _ = env.reset(seed=0)
for _ in range(200):
    obs, *_ = env.step(action)
    view.show(obs)                           # popup window, needs a display
    view.record(obs, obs["camera"]["rgb"])   # or collect frames instead
view.save("out/grasp.mp4")
view.close()
```

`show` and `record` take the whole observation or just the node's sub-dict, plus
`env=` when batched. White is no contact, red is `vmax` (8 mm for `contact_depth`,
2 N for `kinematic`); the scale is fixed, so a colour means the same thing in
every run. Pass `vmax=` to change it.

## Viewing the page over SSH

`out/tactile.html` is self-contained — the SVG and all three renders are inlined,
so it is one ~400 KB file with no sibling assets to fetch. Two ways to see it
from a remote box:

Serve it and let the editor forward the port (VS Code Remote-SSH picks the
listening port up automatically; click through to `http://localhost:8000/tactile.html`):

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory viz/out
```

Or copy the single file down and open it locally — run this on your *local*
machine, substituting your own ssh host:

```bash
scp <host>:/path/to/genesis_adaptor/viz/out/tactile.html ~/Downloads/
```

On plain ssh rather than VS Code, tunnel first with
`ssh -L 8000:localhost:8000 <host>`, then use the same localhost URL.

## Configuration

`common.py` reads `FACE_AXIS`, `ARC`, `N_AROUND`, `N_ALONG`, `MARGIN`, and
`PROBE_RADIUS` from `TactileSensorNodeOptions`, so the visualizations use the
sensor defaults. `build.py` also derives its grid, taxel counts, probe radius,
and area table from the live configuration and artifact output. The scripts remain a
right-hand Wuji visualization with the factory's built-in thumb-axis overrides.

`common.py` is the single place that reaches into the sensor's private helpers
(`_compute_arc_probes`, `_build_link_mesh`, `_FACE_AXES`). If those move, this is the only file
to fix.

## What each script does

| script | output |
|---|---|
| `artifacts.py` | builds the hand and emits the projected face table and three rendered views |
| `renderer.py` | projects and rasterises the hand's collision geometry, then shades taxels with pure NumPy |
| `layout.py` | shared face, tile, connector, bounds, and canvas calculations |
| `bake.py` | compatibility command for building the static artifacts |
| `draw_layout.py` | flat palmar schematic from the face table: every link tiled into an `n_around × n_along` grid, filled edge to edge |
| `build.py` | builds the artifacts and SVG, then fills `template.html` to create the self-contained page |
| `template.html` | page markup, styles, and interactions with placeholders for generated content |
| `tactile_view.py` | live taxel readings painted on the palmar schematic: the hand gray, each pad white through red. A window while an env runs, or an mp4 with the scene camera beside it |
| `check_probes.py` | reports probe distance from the mesh, normal length and direction, and band width |
| `measure_wrap.py` | measures the band that can actually register contact — wider than `arc_degrees`, which only spans probe *centres* |
