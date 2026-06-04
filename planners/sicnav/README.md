# arena_planners_sicnav

SICNav (Samavi, Han, Shkurti, Schoellig — *Safe and Interactive Crowd Navigation
using Model Predictive Control and Bilevel Optimization*, IEEE T-RO 2024) as an
`arena_planners` bridge planner.

The robot trajectory and the predicted crowd motion are optimised jointly: each
human is modelled as an ORCA agent embedded as KKT constraints in a bilevel MPC,
solved with **CasADi + IPOPT** (optionally HSL/MA57). **No acados.**

## Layout

| file | purpose |
|------|---------|
| `planner.py` | bridge entry point: `step(features) -> [v, omega]` |
| `planner.yaml` | datasources: `robot_pose` (TF), `robot_state` (odom), `pedestrians`, `goal_pose`, `global_plan` |
| `configs/policy.config` | SICNav config (`priviledged_info=false` → SICNav-np) |
| `pyproject.toml` | per-planner venv deps (Python 3.10) |
| `CMakeLists.txt` | builds the venv + installs Python-RVO2 |
| `deps/Python-RVO2/` | vendored `rvo2` (ORCA) — required by SICNav's `orca_callback` |

## Mode

Runs **SICNav-np** (`priviledged_info=false`). Arena's `pedestrians` observation
provides position + velocity only (no human goals), which is exactly SICNav-np's
input — it estimates each human's goal internally by constant-velocity projection.
True **SICNav-p** additionally needs a human-goal source (not currently published
by Arena); set `priviledged_info=true` in `configs/policy.config` once one exists.

## Reproducible build (fresh clone)

This planner builds automatically with the rest of the workspace — no manual
`arena feature planners add sicnav` is needed (it is a source-tree planner, picked
up by colcon). A clean build does:

1. `git submodule update --init --recursive` (pulls this dir + `deps/Python-RVO2`).
2. `arena update` → `vcs import` of `arena.repos` (brings `ament_cmake_venv_uv`
   with `uv venv --clear`, `nav2_mask_overlay_layer`, etc.) + `arena build`.
3. colcon builds `arena_planners_sicnav`:
   - `uv` creates a **Python 3.10** venv from `pyproject.toml`,
   - installs the original SICNav library (`SugerSenpai/safe-interactive-crowdnav`
     @ pinned commit — byte-identical to `sepsamavi/safe-interactive-crowdnav`,
     but with a complete `packages` list so it installs non-editable),
   - `CMakeLists.txt` then installs vendored `deps/Python-RVO2` into the venv via
     `uv pip install --no-build-isolation` (Cython + system CMake/g++).

To launch (jackal needs its robot deps first):

```bash
arena feature robots add jackal          # one-time: jackal urdf/meshes/deps
arena launch sim:=gazebo world:=map_empty robot:=jackal \
    mobile:=drl mobile.planner:=sicnav mobile.global_planner:=nav2/navfn \
    tm_robots:=explore tm_obstacles:=random headless:=true
```

### Build pins / gotchas (why the versions are exact)

- **Python 3.10**, not SICNav's `3.8.13` pin: the `arena_planners` bridge SDK uses
  `@dataclass(slots=True)` (3.10+). SICNav's code is 3.10-compatible; its pin is
  relaxed via `[[tool.uv.dependency-metadata]]` in `pyproject.toml`.
- **`gym==0.23.1`** (not SICNav's `0.21.0`): 0.21.0 ships invalid PEP 508 metadata
  (`opencv-python>=3.`) that modern resolvers reject. 0.23.1 is import-compatible
  (only `class CrowdSimPlus(gym.Env)` runs at import).
- **Python-RVO2** is installed by `CMakeLists.txt`, not as a normal dependency: it
  has no `pyproject.toml`, imports Cython in `setup.py`, and builds a bundled C++
  lib via CMake, so it needs `--no-build-isolation` against the venv.

## Performance

Solve time grows steeply with the human count on IPOPT/MUMPS (no HSL):
~0.4 s @2 humans, ~1.1 s @3, ~8 s @5. `_MAX_HUMANS` in `planner.py` defaults to
**3** (the closest pedestrians; padded with far/inactive humans when fewer). For
real-time multi-human navigation install **HSL/MA57** — `campc.py` auto-detects it
when the HSL libs are on the library path (free for academics via Coin-HSL).

## Debugging

Set `SICNAV_DEBUG=/path/to/log` in the launch environment to have `planner.py`
append per-step inputs (robot pose, plan length, pedestrian count, goal) and the
chosen action to that file (the bridge consumes the subprocess stdout/stderr).
