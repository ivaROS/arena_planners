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
import time

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
# MPC integration step. CRUCIAL real-time constraint: with obs_policy=latest_only
# the bridge feeds the planner the freshest observation and the planner solves
# back-to-back, so the effective control period == the solve time, and each MPC
# action is held until the next solve completes. For the plan to be consistent,
# _TIME_STEP must match that achievable solve time. At 5 humans the MA57 solve is
# ~0.5s, so _TIME_STEP=0.5 (2Hz) — SICNav's documented crowd operating point.
# (At 0.25s/4Hz the action was applied ~2x its planned duration -> oversteer +
# crawl.) The policy.config dynamics limits (max_speed, max_rot, max_l_acc, ...)
# are physical rates/accelerations that the MPC multiplies by _TIME_STEP itself,
# so they need NO rescaling when _TIME_STEP changes. Drop to 0.25 only alongside
# _MAX_HUMANS<=3 (solve ~0.26s) for 4Hz reactive control with a smaller crowd.
_TIME_STEP: float = 0.50
# The MPC is built for a fixed human count (set on the first solve). We always
# present exactly this many humans: the closest detected pedestrians, padded with
# far-away inactive humans when fewer are seen.
#
# Solve time grows steeply with this. On IPOPT/MUMPS (no HSL) the bilevel ORCA-KKT
# solve is ~0.4s @2, ~1.1s @3, ~8s @5 — 5 is unusable. With HSL/MA57
# (scripts/install_hsl.sh): ~0.19s @2, ~0.26s @3, ~0.40s @4, ~0.50s @5, ~0.69s @6.
# 5 = SICNav's crowd operating point, run at _TIME_STEP=0.5 (2Hz) so the ~0.5s
# solve matches the control period. campc.py caps the IPOPT wall-clock
# (ipopt.max_cpu_time) so the occasional hard solve can't spike to multiple
# seconds and make the robot drive blind. At _TIME_STEP=0.5 the per-step travel is
# pref_speed*dt=0.45m, so horiz=4 already gives 1.8m lookahead (more than horiz=6
# did at 0.25s) — no need for a long horizon. campc.py auto-detects MA57.
_MAX_HUMANS: int = 5
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
# Rotate-to-face uses hysteresis: start turning when the heading error exceeds
# _ALIGN_ENTER, and keep turning (no MPC) until it drops below _ALIGN_EXIT. This
# prevents the align step and the MPC from fighting (which made the robot spin in
# place): the robot turns cleanly to face the target, then the MPC drives.
# Set _ALIGN_ENTER to 0 to disable and use the MPC unconditionally.
_ALIGN_ENTER: float = math.radians(60.0)
_ALIGN_EXIT: float = math.radians(15.0)
# Proportional turn rate: correct the heading error over ~this many seconds
# (clamped to _OMEGA_MAX). Larger = gentler; avoids the overshoot of err/dt.
_ALIGN_TC: float = 0.5
_aligning: bool = False
# Key identifying the reference last generated (goal + whether it follows the plan);
# regenerate when it changes.
_last_ref_key: tuple | None = None
# Don't rotate-to-align when the target is closer than this: heading-to-target is
# ill-defined on top of the goal, and spinning there stops the robot from settling
# (and the task manager from registering arrival). Let the MPC settle instead.
_ALIGN_MIN_DIST: float = 0.6
# Final-approach pose control: SICNav's point-stabilisation reaches the goal
# POSITION but not the goal HEADING. Arena's goto-pose completion check also
# requires the goal yaw (default tol 30 deg) — so once within this position
# distance of the goal, rotate in place to the goal yaw so arrival registers.
_GOAL_POS_TOL: float = 0.5
_GOAL_YAW_TOL: float = math.radians(12.0)

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


def _build_plan_reference(policy: CollisionAvoidMPC, joint_state, plan) -> bool:
    """Set SICNav's reference (ref_poses_all/ref_actions_all) from the Arena global
    plan so path_foll follows navfn's obstacle-avoiding route (instead of a
    straight line). Returns True on success.

    The plan is arc-length resampled to one MPC step of travel (pref_speed*dt) per
    point; generate_traj() rolls the humans forward via ORCA to fill the rest of
    the reference state. Requires mpc_env (i.e. after the first predict()).
    """
    mpc_env = getattr(policy, "mpc_env", None)
    if mpc_env is None:
        return False
    pts = np.asarray(plan, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return False
    pts = pts[:, :2]
    dt = policy.time_step
    spacing = max(mpc_env.pref_speed * dt, 0.05)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])
    if total < 1e-3:
        return False
    steps = max(2, int(np.ceil(total / spacing)))
    samp = np.linspace(0.0, total, steps + 1)
    rx = np.interp(samp, arc, pts[:, 0])
    ry = np.interp(samp, arc, pts[:, 1])
    # Heading from a windowed finite difference along the path, NOT an adjacent-point
    # np.gradient. The global plan is finely sampled and grid-quantised (navfn emits
    # hundreds of points with cm-scale staircase jitter), so adjacent differences give
    # a heading that zig-zags +-45deg; the MPC then tracks that as a +-1 rad/s heading
    # wiggle (and never builds forward speed). A multi-point window low-passes the
    # orientation while keeping the tracked positions exactly on the planned,
    # obstacle-avoiding route. ~0.5 m window.
    n = len(rx)
    L = max(1, int(round(0.5 / spacing)))
    idx = np.arange(n)
    ahead = np.minimum(idx + L, n - 1)
    behind = np.maximum(idx - L, 0)
    rth = np.unwrap(np.arctan2(ry[ahead] - ry[behind], rx[ahead] - rx[behind]))
    om = ((np.diff(rth) + np.pi) % (2 * np.pi) - np.pi) / dt          # (steps,)
    v = np.full(steps, mpc_env.pref_speed)                            # (steps,)
    x_rob = np.vstack([rx, ry, rth])                                  # (3, steps+1)
    u_rob = np.vstack([v, om])                                        # (2, steps)
    ref_x, ref_u = policy.generate_traj(joint_state, steps, x_rob=x_rob, u_rob=u_rob)
    policy.ref_poses_all = ref_x
    policy.ref_actions_all = ref_u
    return True


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

    theta = float(robot_pose[2])

    # MPC goal: the actual goal pose. path_foll builds a straight-line reference
    # from the robot to this goal and follows it (regenerated on goal change below).
    goal = features.get("goal_pose")
    global_plan = features.get("global_plan")
    goal_xy: tuple[float, float] | None = None
    if goal is not None and len(goal) >= 2:
        goal_xy = (float(goal[0]), float(goal[1]))
    elif global_plan is not None and len(global_plan) > 0:
        gp_arr = np.asarray(global_plan)
        goal_xy = (float(gp_arr[-1][0]), float(gp_arr[-1][1]))
    if goal_xy is None:
        _dbg("no goal -> [0,0]")
        return [0.0, 0.0]

    # Final-approach pose control: once within position tolerance of the goal, stop
    # translating and rotate to the goal heading so the task manager's goto-pose
    # check (position AND yaw) registers arrival (else explore never issues the
    # next goal — SICNav reaches position but not heading).
    if math.hypot(goal_xy[0] - rx, goal_xy[1] - ry) < _GOAL_POS_TOL:
        if goal is not None and len(goal) >= 3:
            yaw_err = (float(goal[2]) - theta + math.pi) % (2 * math.pi) - math.pi
            if abs(yaw_err) > _GOAL_YAW_TOL:
                omega = float(np.clip(yaw_err / _ALIGN_TC, -_OMEGA_MAX, _OMEGA_MAX))
                _dbg(f"final-yaw: err={yaw_err:.2f} -> rotate [0, {omega:.3f}]")
                return [0.0, omega]
        _dbg("at goal pose -> [0,0]")
        return [0.0, 0.0]

    # Unicycle alignment: rotate in place to face the goal before driving, so the
    # MPC starts well-aligned. Hysteretic + proportional (turns cleanly then hands
    # off). Pure rotation, no translation, so it's safe around humans.
    global _aligning
    heading_err = (math.atan2(goal_xy[1] - ry, goal_xy[0] - rx) - theta + math.pi) % (2 * math.pi) - math.pi
    if _ALIGN_ENTER > 0.0:
        if _aligning:
            _aligning = abs(heading_err) > _ALIGN_EXIT
        elif abs(heading_err) > _ALIGN_ENTER:
            _aligning = True
        if _aligning:
            omega = float(np.clip(heading_err / _ALIGN_TC, -_OMEGA_MAX, _OMEGA_MAX))
            _dbg(f"align: heading_err={heading_err:.2f} -> rotate [0, {omega:.3f}]")
            return [0.0, omega]

    robot = _robot_full_state(robot_pose, robot_state, goal_xy)
    humans = _human_states(features.get("pedestrians"), (rx, ry))
    joint_state = FullyObservableJointState(self_state=robot, human_states=humans, static_obs=[])

    # path_foll reference: prefer the Arena global plan (navfn's obstacle-avoiding
    # route); fall back to SICNav's straight-line gen_ref_traj until a plan for the
    # current goal is available. Rebuild only when the goal changes or the plan
    # first becomes available (path_foll tracks the fixed reference via closest
    # point as the robot advances). Needs mpc_env (after the first predict()).
    global _last_ref_key
    if getattr(_policy, "mpc_env", None) is not None:
        plan_ok = (
            global_plan is not None
            and len(global_plan) >= 2
            and math.hypot(
                float(np.asarray(global_plan)[-1][0]) - goal_xy[0],
                float(np.asarray(global_plan)[-1][1]) - goal_xy[1],
            ) < 1.0
        )
        ref_key = (round(goal_xy[0], 1), round(goal_xy[1], 1), bool(plan_ok))
        if ref_key != _last_ref_key:
            built = False
            if plan_ok:
                try:
                    built = _build_plan_reference(_policy, joint_state, global_plan)
                except Exception as exc:
                    _dbg(f"plan reference failed: {exc!r}")
            if not built:
                try:
                    _policy.gen_ref_traj(joint_state)  # straight-line fallback
                    built = True
                except Exception as exc:
                    _dbg(f"gen_ref_traj failed: {exc!r}")
            if built:
                _last_ref_key = ref_key
                _dbg(f"regen ref -> goal {[round(g, 2) for g in goal_xy]} from_plan={plan_ok}")
    _dbg(f"goal={[round(g, 2) for g in goal_xy]} robot v=({round(robot.vx, 3)},{round(robot.vy, 3)}) "
         f"closest_hum={[(round(h.px, 2), round(h.py, 2)) for h in humans[:3]]}")

    try:
        t_solve = time.perf_counter()
        action = _policy.predict(joint_state)
        solve_s = time.perf_counter() - t_solve
    except Exception as exc:  # solver/parse failure -> stop safely
        _log.warning("SICNav predict failed: %s", exc)
        _dbg(f"PREDICT EXCEPTION: {exc!r}")
        return [0.0, 0.0]

    # CAMPC returns ActionRot(v, omega*dt) for the unicycle base.
    v = float(np.clip(action.v, -_V_MAX, _V_MAX))
    omega = float(np.clip(action.r / _TIME_STEP, -_OMEGA_MAX, _OMEGA_MAX))
    solved = _policy.mpc_sol_succ[-1] if getattr(_policy, "mpc_sol_succ", None) else "?"
    # Solver diagnostics: campc reports solved=True even when it DISCARDS the solve
    # and returns the warmstart guess (final_obj > init_obj) or hits an iter/time
    # limit. status (2=Succeeded, 1=Acceptable, -2=MaxIter), iter_count, and the
    # debug_text ("Solution worse than warmstart" / "USING WARMSTART GUESS") tell us
    # which — key to diagnosing the conservative cruise speed.
    _ss = getattr(_policy, "solver_summary", None)
    status = _ss["optim_status"][-1] if _ss and _ss.get("optim_status") else "?"
    iters = _ss["iter_count"][-1] if _ss and _ss.get("iter_count") else "?"
    dtext = _policy.all_debug_text[-1] if getattr(_policy, "all_debug_text", None) else "?"
    dist_goal = math.hypot(goal_xy[0] - rx, goal_xy[1] - ry)
    _dbg(f"action raw v={action.v:.4f} r={action.r:.4f} solved={solved} status={status} iters={iters} "
         f"why={dtext!r} solve_s={solve_s:.3f} dist_goal={dist_goal:.2f} -> [v={v:.4f}, omega={omega:.4f}]")
    return [v, omega]


def on_reset(episode_id: str, initial_state: dict | None) -> None:
    # Rebuild the MPC for the next episode (re-fixes human count, resets warmstart).
    global _policy, _aligning, _last_ref_key
    _policy = None
    _aligning = False
    _last_ref_key = None


if __name__ == "__main__":
    manifest = load_manifest(pathlib.Path(__file__).parent / "planner.yaml")
    main_loop(step, manifest=manifest, on_reset=on_reset)
