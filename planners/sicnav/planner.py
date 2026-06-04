"""SICNav wrapper for the arena_planners bridge.

SICNav (Samavi, Han, Shkurti, Schoellig — "Safe and Interactive Crowd
Navigation using Model Predictive Control and Bilevel Optimization", T-RO 2024)
is a local planner that jointly optimises the robot trajectory and the predicted
crowd motion, modelling each human as an ORCA agent embedded as KKT constraints
in a bilevel MPC. It is solved with CasADi + IPOPT (optionally HSL/MA57); it does
NOT use acados.

Policy library: the original SICNav code, installed from
``SugerSenpai/safe-interactive-crowdnav`` which mirrors the original
``sepsamavi/safe-interactive-crowdnav`` campc.py byte-for-byte but ships a
complete ``packages`` list (the original setup.py omits ``sicnav.utils.mpc_utils``
and so cannot be pip-installed non-editable).

Contract: ``step(features) -> [v, omega]``.

Mode: SICNav-np (``priviledged_info = false``). Arena's pedestrian observations
provide position + velocity only (no human goals), which is exactly the input
SICNav-np expects — it estimates each human's goal internally by constant-velocity
projection. True SICNav-p would additionally need a human-goal datasource.
"""

from __future__ import annotations

import configparser
import logging
import math
import os
import pathlib

import numpy as np
from arena_planners.geometry import lookahead_on_path
from arena_planners.sdk import load_manifest, main_loop

from crowd_sim_plus.envs.utils.state_plus import FullState, FullyObservableJointState
from sicnav.policy.campc import CollisionAvoidMPC

_log = logging.getLogger("arena_planners.sicnav")

# Opt-in file debug (SICNAV_DEBUG=/path): the bridge consumes the subprocess
# stdout/stderr, so log key per-step data to a file we can read directly.
_DEBUG_PATH = os.environ.get("SICNAV_DEBUG")


def _dbg(msg: str) -> None:
    if not _DEBUG_PATH:
        return
    try:
        with open(_DEBUG_PATH, "a") as fh:
            fh.write(msg + "\n")
    except Exception:
        pass

# Must match the dynamics assumptions baked into configs/policy.config.
_TIME_STEP: float = 0.25
# The MPC is built for a fixed human count (set on the first solve). We always
# present exactly this many humans: the closest detected pedestrians, padded with
# far-away inactive humans when fewer are seen.
#
# Solve time grows steeply with this (IPOPT/MUMPS, no HSL): ~0.4s @2, ~1.1s @3,
# ~8s @5. Original SICNav reaches real-time with the HSL/MA57 linear solver; until
# that's installed, keep this small. campc.py auto-detects MA57 and uses it if the
# HSL libs are on the library path.
_MAX_HUMANS: int = 3
_ROBOT_RADIUS: float = 0.3
_HUMAN_RADIUS: float = 0.3
_V_PREF: float = 0.9          # = pref_speed in policy.config
_V_MAX: float = 0.95          # = max_speed in policy.config
_OMEGA_MAX: float = math.radians(60.0) / _TIME_STEP  # = max_rot_degrees / dt
_LOOKAHEAD: float = 3.0       # path lookahead for the MPC stabilisation point
_HUMAN_GOAL_PROJ: float = 2.0  # seconds of constant-velocity goal projection
_FAR: float = 1.0e3           # placement offset for padding humans
# Heading error (rad) beyond which we rotate in place toward the target before
# handing off to the MPC. SICNav's point-stabilisation MPC has a symmetric
# zero-gradient equilibrium when the goal is ~180 deg behind the robot and stalls
# there; this alignment step (cf. nav2's rotation shim) breaks that symmetry.
# Set to 0 to disable and use the MPC unconditionally.
_ALIGN_THRESHOLD: float = math.radians(100.0)
# Don't rotate-to-align when the target is closer than this: heading-to-target is
# ill-defined on top of the goal, and spinning there stops the robot from settling
# (and the task manager from registering arrival). Let the MPC settle instead.
_ALIGN_MIN_DIST: float = 0.6

_policy: CollisionAvoidMPC | None = None


class _Env:
    """Minimal env shim. CAMPC reads ``time_step``, ``global_time``, ``sim_env``
    and ``config`` (to build its dummy/template human)."""

    def __init__(self, time_step: float) -> None:
        self.time_step = time_step
        self.time_limit = 100.0
        # Non-zero so predict() does not re-initialise the MPC on every call
        # (it re-inits only when mpc_env is None or global_time == 0.0).
        self.global_time = 1.0
        self.sim_env = "general"
        self.config = configparser.ConfigParser()
        self.config.read_dict(
            {
                "humans": {
                    "visible": "true",
                    "radius": str(_HUMAN_RADIUS),
                    "v_pref": "1.0",
                    "safety_space": "0.01",
                    "policy": "orca_plus",
                    "sensor": "coordinates",
                },
                "env": {"SB3": "False"},
            }
        )

    def set_human_observability(self, priviledged_info: bool) -> None:
        # Hook called by CAMPC.set_env; nothing to toggle in this headless shim.
        pass


def _build_policy() -> CollisionAvoidMPC:
    config_path = pathlib.Path(__file__).parent / "configs" / "policy.config"
    cfg = configparser.RawConfigParser()
    cfg.read(str(config_path))
    policy = CollisionAvoidMPC()
    policy.configure(cfg)          # sets priviledged_info, horizon, etc.
    policy.set_phase("test")
    policy.set_device("cpu")
    policy.set_env(_Env(_TIME_STEP))  # after configure: builds dummy human w/ priviledged flag
    _log.info("SICNav CAMPC initialised (priviledged_info=%s, horiz=%s)", policy.priviledged_info, policy.horiz)
    return policy


def _robot_full_state(robot_pose, robot_state, goal_xy) -> FullState:
    px, py, theta = float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])
    # robot_state = [x, y, vx, vy, theta, omega]; odom twist is body-frame for a
    # diff-drive base (linear.x = forward speed, linear.y ~ 0). SICNav requires the
    # world-frame velocity to be consistent with heading (vx=v*cos, vy=v*sin), so
    # rebuild it from forward speed + heading.
    v_forward = 0.0
    if robot_state is not None and len(robot_state) > 2:
        v_forward = float(robot_state[2])
    vx = v_forward * math.cos(theta)
    vy = v_forward * math.sin(theta)
    gx, gy = goal_xy
    return FullState(px, py, vx, vy, _ROBOT_RADIUS, gx, gy, _V_PREF, theta)


def _human_states(peds, robot_xy) -> list[FullState]:
    """Exactly _MAX_HUMANS humans: closest detected pedestrians + far padding.

    Each human carries a constant-velocity goal projection so the state is valid
    in both SICNav-np (goals re-estimated internally) and -p (goals used directly)
    code paths.
    """
    rx, ry = robot_xy
    detected: list[FullState] = []
    if peds is not None:
        rows = [tuple(float(c) for c in row) for row in peds]
        rows.sort(key=lambda r: (r[1] - rx) ** 2 + (r[2] - ry) ** 2)
        for _id, x, y, vx, vy in rows[:_MAX_HUMANS]:
            gx = x + vx * _HUMAN_GOAL_PROJ
            gy = y + vy * _HUMAN_GOAL_PROJ
            theta = math.atan2(vy, vx) if (vx or vy) else 0.0
            detected.append(FullState(x, y, vx, vy, _HUMAN_RADIUS, gx, gy, _V_PREF, theta))

    for i in range(len(detected), _MAX_HUMANS):
        # Far, static, distinct positions -> ORCA constraints stay inactive.
        fx, fy = rx + _FAR + i, ry + _FAR + i
        detected.append(FullState(fx, fy, 0.0, 0.0, _HUMAN_RADIUS, fx, fy, _V_PREF, 0.0))
    return detected


def step(features: dict) -> list[float]:
    global _policy
    if _policy is None:
        _policy = _build_policy()

    _dbg(f"--- step: keys={sorted(features.keys())}")
    robot_pose = features.get("robot_pose")
    if robot_pose is None or len(robot_pose) < 3:
        _dbg(f"no robot_pose -> [0,0]; robot_pose={robot_pose}")
        return [0.0, 0.0]
    robot_state = features.get("robot_state")
    rx, ry = float(robot_pose[0]), float(robot_pose[1])
    peds_in = features.get("pedestrians")
    n_peds = 0 if peds_in is None else len(peds_in)
    gp = features.get("global_plan")
    n_plan = 0 if gp is None else len(gp)
    _dbg(f"robot_pose={[round(float(v),3) for v in robot_pose[:3]]} n_plan={n_plan} n_peds={n_peds} goal_pose={features.get('goal_pose')}")

    # Stabilisation point: lookahead along the global plan, else the goal pose.
    target = None
    global_plan = features.get("global_plan")
    if global_plan is not None and len(global_plan) > 0:
        target = lookahead_on_path(np.asarray(global_plan), np.asarray(robot_pose), lookahead=_LOOKAHEAD)
    if target is None:
        goal_pose = features.get("goal_pose")
        if goal_pose is not None and len(goal_pose) >= 2:
            target = (float(goal_pose[0]), float(goal_pose[1]))
    if target is None:
        _dbg("no target -> [0,0]")
        return [0.0, 0.0]

    # Unicycle alignment: if the target is far off the robot's heading, rotate in
    # place toward it first (SICNav's point_stab MPC stalls at the ~180 deg
    # equilibrium). Pure rotation, no translation, so it's safe around humans.
    theta = float(robot_pose[2])
    dist_to_target = math.hypot(target[0] - rx, target[1] - ry)
    heading_err = (math.atan2(target[1] - ry, target[0] - rx) - theta + math.pi) % (2 * math.pi) - math.pi
    if _ALIGN_THRESHOLD > 0.0 and dist_to_target > _ALIGN_MIN_DIST and abs(heading_err) > _ALIGN_THRESHOLD:
        omega = float(np.clip(heading_err / _TIME_STEP, -_OMEGA_MAX, _OMEGA_MAX))
        _dbg(f"align: heading_err={heading_err:.2f} -> rotate [0, {omega:.3f}]")
        return [0.0, omega]

    robot = _robot_full_state(robot_pose, robot_state, target)
    humans = _human_states(features.get("pedestrians"), (rx, ry))
    _dbg(f"target={[round(t,3) for t in target]} robot v=({round(robot.vx,3)},{round(robot.vy,3)}) "
         f"closest_hum={[(round(h.px,2),round(h.py,2)) for h in humans[:3]]}")
    joint_state = FullyObservableJointState(self_state=robot, human_states=humans, static_obs=[])

    try:
        action = _policy.predict(joint_state)
    except Exception as exc:  # solver/parse failure -> stop safely
        _log.warning("SICNav predict failed: %s", exc)
        _dbg(f"PREDICT EXCEPTION: {exc!r}")
        return [0.0, 0.0]

    # CAMPC returns ActionRot(v, omega*dt) for the unicycle base.
    v = float(np.clip(action.v, -_V_MAX, _V_MAX))
    omega = float(np.clip(action.r / _TIME_STEP, -_OMEGA_MAX, _OMEGA_MAX))
    _dbg(f"action raw v={action.v:.4f} r={action.r:.4f} -> [v={v:.4f}, omega={omega:.4f}]")
    return [v, omega]


def on_reset(episode_id: str, initial_state: dict | None) -> None:
    # Rebuild the MPC for the next episode (re-fixes human count, resets warmstart).
    global _policy
    _policy = None


if __name__ == "__main__":
    manifest = load_manifest(pathlib.Path(__file__).parent / "planner.yaml")
    main_loop(step, manifest=manifest, on_reset=on_reset)
