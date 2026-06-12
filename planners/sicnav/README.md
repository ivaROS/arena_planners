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

## Performance & HSL/MA57

SICNav's bilevel ORCA-KKT MPC is solved by IPOPT, and the linear solver dominates
the per-step cost. `_MAX_HUMANS` in `planner.py` caps how many pedestrians enter
the MPC (the closest ones; padded with far/inactive humans when fewer are seen).
The MPC step period is 0.25 s. Measured solve time per step:

| humans | IPOPT/MUMPS (default) | IPOPT/MA57 (HSL) |
|-------:|----------------------:|-----------------:|
|      2 | ~0.4 s                | ~0.19 s          |
|      3 | ~1.1 s                | ~0.26 s          |
|      4 | —                     | ~0.40 s          |
|      5 | ~8 s                  | ~0.50 s          |
|      6 | —                     | ~0.69 s          |

Without HSL, only ~3 humans is workable; 5 is effectively unusable (~8 s/step).
With **HSL/MA57** the solve is ~16× faster at 5 humans, so `_MAX_HUMANS` defaults
to **5** (the original SICNav crowd operating point, ~2 Hz). `campc.py`
auto-detects MA57 at startup and uses it whenever the HSL library is on the
planner venv's library path. Drop `_MAX_HUMANS` back to 3 for the tightest
(~4 Hz) control rate, or if you run without HSL.

### Installing HSL/MA57

HSL (Coin-HSL) is free for academics but distributed under a **per-user licence**,
so it **cannot be committed to this repo or shared between collaborators** — every
person who wants MA57 obtains their own licence and supplies their own source
tarball. The steps below take a collaborator from "no licence" to a working MA57
build. `scripts/install_hsl.sh` automates everything after the download.

1. **Apply for the licence.** Go to <https://licences.stfc.ac.uk/product/coin-hsl>,
   create an account with your **academic/institutional email**, and request the
   **Coin-HSL** package (the full one — it includes MA57 — *not* the smaller
   "Coin-HSL Archive"). Approval is manual and usually lands by email within a day
   or two. You only need to do this once.
2. **Download the source** once approved. From your STFC account, download the
   Coin-HSL **source tarball** (e.g. `coinhsl-2024.05.15.tar.gz`). Put it somewhere
   the Arena container can read — the installer searches `~/arena_ws/hsl/` and
   `<workspace>/build/hsl_build/` by default, or you can pass an explicit path.
3. **Build the SICNav planner first** (`arena build arena_planners_sicnav`) so its
   venv exists for the installer to wire HSL into.
4. **Run the installer inside the Arena container**, pointing it at your tarball:
   ```sh
   docker exec -u 0 <arena-container> bash -lc \
     'bash /opt/arena_ws/src/Arena/arena_planners/planners/sicnav/scripts/install_hsl.sh \
      /opt/arena_ws/build/hsl_build/coinhsl-2024.05.15.tar.gz'
   ```
   (find `<arena-container>` with `docker ps`; it's typically `arena-<workspace>-arena-1`).
   The script installs build deps (gfortran, meson, ninja, libmetis-dev), compiles
   `libcoinhsl.so`, and installs a self-contained `libhsl.so` (with `libmetis.so.5`
   bundled and `RPATH=$ORIGIN`) into the venv's `casadi/` package dir — which is
   where IPOPT's `dlopen("libhsl.so")` looks (its `libipopt.so` has `RPATH=$ORIGIN`).
   It verifies MA57 actually loads (`>>> SUCCESS: IPOPT loaded MA57`) before exiting.
5. **Done.** Nothing else to configure — `campc.py` auto-detects MA57 on the next
   run. You can confirm it in the planner logs: `[CAMPC] MA57 linear solver
   available, using it`.

Re-run the installer (step 4) after any clean rebuild of the SICNav venv — a venv
rebuild reinstalls `casadi` and removes the bundled `libhsl.so`. Collaborators
*without* an HSL licence don't need to do anything: SICNav runs on the default
MUMPS solver (keep `_MAX_HUMANS` at 3 in `planner.py`).

## Debugging

Set `SICNAV_DEBUG=/path/to/log` in the launch environment to have `planner.py`
append per-step inputs (robot pose, plan length, pedestrian count, goal) and the
chosen action to that file (the bridge consumes the subprocess stdout/stderr).

## Troubleshooting

**`ModuleNotFoundError: No module named 'msgpack'` / `'zmq'`** (in
`task_generator_node` / `arena_planners.bridge.edge_node` at launch). The bridge
*host* node runs in the main workspace venv (`/opt/venv`), and these are declared
deps of the bridge (`arena_planners/pyproject.toml`: `pyzmq`, `msgpack`,
`msgpack-numpy`). The error means that venv is **stale** — it predates those deps
and hasn't been re-synced. This affects *all* bridge planners (drlvo, crowdnav,
sicnav), not just SICNav. Fix by re-syncing the workspace venv:

```sh
arena pull      # runs `uv sync` for you, or directly inside the container:
cd /opt/arena_ws/src/Arena && UV_PROJECT_ENVIRONMENT=/opt/venv uv sync --inexact --project .
```

A fresh `install.sh` and every `arena pull` already run this sync, so collaborators
on the normal flow get the deps automatically; this only bites a long-lived venv
that skipped a sync after the deps were added upstream.
