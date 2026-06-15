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
# _TIME_STEP must match that achievable solve time. At _MAX_HUMANS=3 the MA57 solve
# is ~0.26s, so _TIME_STEP=0.25 (4Hz) keeps the held action close to its planned
# duration. (5 humans @0.5s/2Hz was the nominal crowd point, but the ~0.5-0.6s solve
# blew the CPU cap ~99% of the time at the crossing -> MPC starved -> stall; see
# _MAX_HUMANS.) Earlier 0.25 attempts oversteered because they ran 5 humans (solve
# ~0.5s >> 0.25s step, action held ~2x); paired with _MAX_HUMANS=3 the solve now
# matches the step. The policy.config dynamics limits (max_speed, max_rot,
# max_l_acc, ...) are physical rates/accelerations the MPC multiplies by _TIME_STEP
# itself, so they need NO rescaling here; _OMEGA_MAX below tracks _TIME_STEP.
_TIME_STEP: float = 0.25
# The MPC is built for a fixed human count (set on the first solve). We always
# present exactly this many humans: the closest detected pedestrians, padded with
# far-away inactive humans when fewer are seen.
#
# Solve time grows steeply with this. On IPOPT/MUMPS (no HSL) the bilevel ORCA-KKT
# solve is ~0.4s @2, ~1.1s @3, ~8s @5 — 5 is unusable. With HSL/MA57
# (scripts/install_hsl.sh): ~0.19s @2, ~0.26s @3, ~0.40s @4, ~0.50s @5, ~0.69s @6.
# WAS 5 @ _TIME_STEP=0.5 (SICNav's nominal crowd point) but at 5 humans the solve
# (~0.5-0.6s) blew the 0.45s CPU cap ~99% of the time on the crossing scenario:
# the MPC almost never returned a timely, trustworthy plan, so recovery/align took
# over and the robot stalled+wiggled instead of negotiating past the crosser. 3 @
# _TIME_STEP=0.25 solves ~0.26s -> converges within budget AND controls at 4Hz, so
# the MPC actually drives the crossing. The MPC now models only the 3 CLOSEST
# pedestrians; any others are still covered by the prediction-independent
# _safety_brake(). campc.py caps the IPOPT wall-clock (ipopt.max_cpu_time, now
# ~0.30s to bound the held action near the 0.25s step) so a hard solve can't spike
# and drive the robot blind. campc.py auto-detects MA57.
_MAX_HUMANS: int = 3
_ROBOT_RADIUS: float = 0.3
_HUMAN_RADIUS: float = 0.3
_V_PREF: float = 0.3          # = pref_speed in policy.config
_V_MAX: float = 0.35          # = max_speed in policy.config
# Physical angular-rate cap = max_rot_degrees(per-step) / _TIME_STEP. max_rot_degrees
# is a PER-STEP rotation bound on the MPC's action.r (mpc_env control bounds), so the
# physical rate scales as 1/_TIME_STEP. Halving _TIME_STEP 0.5->0.25 would double the
# rate to 240 deg/s for the same 60deg/step; we halved max_rot_degrees to 30 (in
# policy.config) to KEEP the well-tuned ~120 deg/s. Keep these two in sync.
_OMEGA_MAX: float = math.radians(30.0) / _TIME_STEP  # = max_rot_degrees / dt = ~120 deg/s
_LOOKAHEAD: float = 3.0       # path lookahead for the MPC stabilisation point
_HUMAN_GOAL_PROJ: float = 2.0  # seconds of constant-velocity goal projection
_FAR: float = 1.0e3           # placement offset for padding humans
# On a failed solve (no collision-checked trajectory) the recovery only creeps
# forward if the closest pedestrian is beyond this; nearer than this it holds (v=0)
# and waits for a converged solve, so a failure never drives at anyone.
_RECOVERY_CLEAR_DIST: float = 2.0
# Gentle turn-rate cap for failed-solve recovery when a pedestrian is near. The old
# recovery turned at up to _OMEGA_MAX toward the (jittery, ped-avoidance) path
# heading, which spun the robot in place at a crossing. When blocked we instead turn
# slowly toward the STABLE goal bearing, capped here, so the robot holds and waits
# for a gap facing roughly the goal rather than whirling.
_REC_OMEGA_CAP: float = math.radians(45.0)
# Reactive safety brake (prediction-INDEPENDENT). SICNav's MPC enforces collision
# avoidance against its INTERNAL human prediction (reciprocal-ORCA + constant-velocity
# goal projection). Arena's pedestrians move by social forces and do NOT avoid
# reciprocally, so that prediction can be wrong and the MPC drives a "converged, safe"
# trajectory straight through the real pedestrian (forced-crossing test: min 0.19m
# center-to-center, well inside the 0.6m sum-of-radii; a 5x larger MPC keep-out margin
# did not help -> it's the prediction, not the constraint). This brake is a hard safety
# net ON TOP of the MPC, using MEASURED pedestrian positions/velocities only: scale the
# commanded forward speed down as the robot CLOSES on the nearest pedestrian, to a full
# stop at _BRAKE_STOP_DIST. It caps speed ONLY when the commanded motion reduces
# clearance, so the robot resumes the moment the pedestrian passes or moves away.
_BRAKE_SLOW_DIST: float = 1.5   # center-to-center distance at which to begin slowing
_BRAKE_STOP_DIST: float = 0.8   # full stop (0.6m sum-of-radii + 0.2m margin)
# Anti-freeze maneuver (_unfreeze): break a reciprocal deadlock when a social-force
# pedestrian sits on the robot's path and SICNav's ORCA model keeps waiting for it to
# yield. A freeze = commanded |v| < _FREEZE_V_EPS while the nearest ped is within
# _FREEZE_PED_DIST; sustaining that for _FREEZE_TRIGGER steps escalates to a slow
# creep at _GOAROUND_SPEED that FOLLOWS THE GLOBAL PLAN (peds imprint on the nav2
# costmap, so navfn already routes the plan around them — verified live). The robot
# steers toward the global-plan lookahead point (_GOAROUND_LOOKAHEAD ahead) and creeps
# with a tighter _GOAROUND_STOP_DIST brake margin (still > 0.6m sum-of-radii). If a ped
# is nearer than _GOAROUND_MIN_ARC_DIST in the travel direction (within _PATH_HALF_WIDTH
# laterally) the robot is too close to move forward, so it first backs up — only if the
# rear is clear of peds within _GOAROUND_REAR_CLEAR. The maneuver ends once no ped is
# within _FREEZE_PED_DIST. Step counts derive from _TIME_STEP (4Hz).
_FREEZE_V_EPS: float = 0.05
_FREEZE_PED_DIST: float = 1.6
_FREEZE_TRIGGER: int = max(1, round(3.0 / _TIME_STEP))   # ~3s frozen before escalating
_GOAROUND_SPEED: float = 0.15
_GOAROUND_LOOKAHEAD: float = 1.5                         # lookahead along the global plan
_GOAROUND_FACE_TOL: float = math.radians(25.0)           # creep only when ~facing the aim
_GOAROUND_STOP_DIST: float = 0.7                         # 0.6 sum-of-radii + 0.1 margin
_GOAROUND_MIN_ARC_DIST: float = 1.0                      # nearer ahead than this -> back up first
_GOAROUND_REAR_CLEAR: float = 0.9                        # rear must be clear within this to back up
_PATH_HALF_WIDTH: float = 0.75                           # lateral half-width of the "blocking" corridor
_frozen_steps: int = 0
_goaround_active: bool = False
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
# Suppress the in-place align spin when a pedestrian is within this distance. Near a
# pedestrian, navfn re-routes around the moving ped every cycle, so the global-plan
# lookahead heading (what align/recovery chase via _drive_heading_err) swings wildly
# — at a crossing it swung ~110 deg and the robot rotated in place chasing it (the
# "wiggle") instead of crossing, never letting the MPC drive. Within this radius we
# skip the align spin and defer heading to the (fast, 4Hz) MPC, and the failed-solve
# recovery turns toward the STABLE goal bearing rather than the jittery path heading.
_ALIGN_PED_GATE: float = 2.0
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
# Must be < the task_generator goal_radius (default 0.3m, staged/impl.py) or the
# robot declares "at goal" and stops driving while still outside the scoring
# circle, so the episode never completes (same goal re-published forever).
_GOAL_POS_TOL: float = 0.25
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


def _drive_heading_err(global_plan, robot_pose, goal_xy) -> float:
    """Heading error (rad) from the robot to the direction it SHOULD drive: the
    global-plan lookahead point if a plan is available, else the straight-line goal.

    The rotate-to-align step and the failed-solve recovery both use this. Aligning to
    the PATH lookahead heading (what the path_foll MPC tracks) rather than the goal
    bearing is what stops them thrashing: when navfn's route initially leaves on a
    different bearing than robot->goal, aligning to the goal would point the robot off
    the path, the MPC would immediately steer back onto it, and the rotate step would
    re-fire — in-place rotation cycling against forward motion."""
    rx, ry, th = float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])
    tx, ty = goal_xy
    if global_plan is not None and len(global_plan) >= 2:
        la = lookahead_on_path(
            np.asarray(global_plan, dtype=float), np.asarray(robot_pose, dtype=float), _LOOKAHEAD
        )
        if la is not None:
            tx, ty = la
    return (math.atan2(ty - ry, tx - rx) - th + math.pi) % (2 * math.pi) - math.pi


def _safety_brake(v: float, robot_pose, humans,
                  slow_dist: float = _BRAKE_SLOW_DIST,
                  stop_dist: float = _BRAKE_STOP_DIST) -> float:
    """Prediction-independent reactive speed cap (see the _BRAKE_* constants).

    Reduce the commanded forward speed as the robot closes on the nearest MEASURED
    pedestrian, to a full stop at stop_dist. This is a hard safety net beneath
    SICNav's MPC, which only avoids its internal (reciprocal-ORCA) human prediction and
    can therefore drive a converged trajectory through a real social-force pedestrian.

    Only the robot's OWN approach is braked: speed is capped only when the commanded
    motion would reduce clearance to a pedestrian (closing speed > 0), so the robot
    proceeds again as soon as the pedestrian crosses past or moves away. Uses measured
    positions/velocities only — completely independent of the MPC's prediction.

    slow_dist/stop_dist default to the cruise margins; the deliberate go-around
    maneuver (_unfreeze) passes a tighter stop_dist so it can pass a path-blocking
    pedestrian at a closer (but still > sum-of-radii) tangential clearance."""
    if v <= 0.0:
        return v  # reversing / already stopped never drives forward into a human
    rx, ry, th = float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])
    vrx, vry = v * math.cos(th), v * math.sin(th)
    v_cap = _V_MAX
    engaged: tuple[float, float] | None = None
    for h in humans:
        dx, dy = float(h.px) - rx, float(h.py) - ry
        d = math.hypot(dx, dy)
        if d >= slow_dist:
            continue
        nx, ny = dx / d, dy / d
        # Closing speed of the robot onto this human: n.(v_robot - v_human), where n
        # points robot->human. >0 means the commanded motion is reducing clearance.
        # Gate the whole brake (including the hard stop) on closing>0: a pedestrian
        # the robot is moving AWAY from (e.g. behind it, or one it has just passed)
        # must NOT hard-stop forward motion — otherwise a ped lingering within
        # stop_dist behind the robot freezes it even though driving on is safe.
        closing = (vrx - float(h.vx)) * nx + (vry - float(h.vy)) * ny
        if closing <= 0.0:
            continue
        if d <= stop_dist:
            v_cap, engaged = 0.0, (d, 0.0)
            break
        cap = (d - stop_dist) / (slow_dist - stop_dist) * _V_MAX
        if cap < v_cap:
            v_cap, engaged = cap, (d, cap)
    if engaged is not None and v_cap < v:
        _dbg(f"SAFETY BRAKE: v {v:.3f} -> {v_cap:.3f} (nearest engaged d={engaged[0]:.2f})")
        return v_cap
    return v


def _unfreeze(v: float, omega: float, robot_pose, humans, goal_xy, global_plan) -> list[float]:
    """Deadlock-breaking maneuver (prediction-independent, beneath the MPC).

    SICNav's ORCA human model assumes pedestrians reciprocally avoid, so when a
    social-force pedestrian instead sits on the robot's path the MPC (and the safety
    brake) hold the robot at a standstill expecting it to move — they mutually freeze
    and the robot never reaches its goal. This detects a SUSTAINED freeze (commanded
    ~0 while a ped is close) and escalates to a slow creep that FOLLOWS THE GLOBAL
    PLAN. Pedestrians imprint as lethal obstacles (lidar) with inflation on the nav2
    costmap, so navfn already routes the global plan around them — verified live:
    ped cells read 100 in both obstacle_layer and global_costmap. The deadlock is
    purely that the local MPC won't execute that already-safe route. So when blocked
    we steer toward the global-plan lookahead point and creep along it with a tighter
    (still > sum-of-radii) brake margin. If a ped is too close in the travel direction
    to move forward, back up first (when the rear is clear) to open room. Normal
    momentary yielding is untouched (only freezes past _FREEZE_TRIGGER escalate); the
    maneuver ends once no ped is within _FREEZE_PED_DIST."""
    global _frozen_steps, _goaround_active
    rx, ry, th = float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])

    nearest = min((math.hypot(float(h.px) - rx, float(h.py) - ry) for h in humans), default=_FAR)
    if nearest > _FREEZE_PED_DIST:
        # Fully clear -> normal command; reset state (this is also the maneuver exit).
        _frozen_steps = 0
        _goaround_active = False
        return [v, omega]

    if not _goaround_active:
        # Count consecutive stopped-AND-near-a-ped steps; escalate past the trigger.
        if abs(v) < _FREEZE_V_EPS:
            _frozen_steps += 1
        else:
            _frozen_steps = 0
        if _frozen_steps < _FREEZE_TRIGGER:
            return [v, omega]
        _goaround_active = True
        _dbg(f"UNFREEZE: escalating (frozen {_frozen_steps} steps, nearest={nearest:.2f})")

    # --- active maneuver: steer along the global plan (already routed around peds) --
    aim = math.atan2(goal_xy[1] - ry, goal_xy[0] - rx)  # fallback: straight at goal
    if global_plan is not None and len(global_plan) >= 2:
        la = lookahead_on_path(
            np.asarray(global_plan, dtype=float), np.asarray(robot_pose, dtype=float),
            _GOAROUND_LOOKAHEAD,
        )
        if la is not None:
            aim = math.atan2(la[1] - ry, la[0] - rx)
    ax, ay = math.cos(aim), math.sin(aim)
    herr = (aim - th + math.pi) % (2 * math.pi) - math.pi

    # Ped too close in the travel (aim) direction to move forward? Back up to open
    # room first — but only if the rear (opposite the aim) is clear, else we'd reverse
    # into someone. Rotate toward the aim while backing so we're oriented to drive out.
    blocker_d = _FAR
    for h in humans:
        dx, dy = float(h.px) - rx, float(h.py) - ry
        d = math.hypot(dx, dy)
        if d <= _FREEZE_PED_DIST and (dx * ax + dy * ay) > 0.0 and abs(dx * -ay + dy * ax) < _PATH_HALF_WIDTH:
            blocker_d = min(blocker_d, d)
    if blocker_d < _GOAROUND_MIN_ARC_DIST:
        rear_clear = all(
            not (math.hypot(float(h.px) - rx, float(h.py) - ry) <= _GOAROUND_REAR_CLEAR
                 and ((float(h.px) - rx) * ax + (float(h.py) - ry) * ay) < 0.0)
            for h in humans
        )
        if rear_clear:
            bk_omega = float(np.clip(herr / _ALIGN_TC, -_OMEGA_MAX, _OMEGA_MAX))
            _dbg(f"UNFREEZE: backing up to open room (blocker_d={blocker_d:.2f}) "
                 f"[v={-_GOAROUND_SPEED:.3f}, omega={bk_omega:.3f}]")
            return [-_GOAROUND_SPEED, bk_omega]
        _dbg(f"UNFREEZE: sandwiched (blocker_d={blocker_d:.2f}, rear blocked) -> hold")

    ga_omega = float(np.clip(herr / _ALIGN_TC, -_OMEGA_MAX, _OMEGA_MAX))
    ga_v = _GOAROUND_SPEED if abs(herr) < _GOAROUND_FACE_TOL else 0.0
    ga_v = _safety_brake(ga_v, robot_pose, humans, stop_dist=_GOAROUND_STOP_DIST)
    _dbg(f"UNFREEZE: follow-plan [v={ga_v:.3f}, omega={ga_omega:.3f}] "
         f"(aim={aim:.2f} herr={herr:.2f} nearest={nearest:.2f} blocker_d={blocker_d:.2f})")
    return [ga_v, ga_omega]


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

    robot = _robot_full_state(robot_pose, robot_state, goal_xy)
    humans = _human_states(features.get("pedestrians"), (rx, ry))
    joint_state = FullyObservableJointState(self_state=robot, human_states=humans, static_obs=[])
    nearest_hum = min((math.hypot(h.px - rx, h.py - ry) for h in humans), default=_FAR)

    # Unicycle alignment: rotate in place to face the goal before driving, so the
    # MPC starts well-aligned. Hysteretic + proportional (turns cleanly then hands
    # off). Pure rotation, no translation, so it's safe around humans. GATED on
    # pedestrian proximity (_ALIGN_PED_GATE): within that radius the global-plan
    # lookahead heading swings as navfn re-routes around the moving ped, so chasing
    # it spins the robot in place (the crossing "wiggle"); there we skip align and
    # let the MPC drive the crossing directly.
    global _aligning
    heading_err = _drive_heading_err(global_plan, robot_pose, goal_xy)
    if _ALIGN_ENTER > 0.0 and nearest_hum > _ALIGN_PED_GATE:
        if _aligning:
            _aligning = abs(heading_err) > _ALIGN_EXIT
        elif abs(heading_err) > _ALIGN_ENTER:
            _aligning = True
        if _aligning:
            omega = float(np.clip(heading_err / _ALIGN_TC, -_OMEGA_MAX, _OMEGA_MAX))
            _dbg(f"align: heading_err={heading_err:.2f} -> rotate [0, {omega:.3f}]")
            return [0.0, omega]
    else:
        _aligning = False  # don't carry align state into a near-ped encounter

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

    # A non-converged solve (status not in {Solved=2, Acceptable=1} — e.g. -5
    # Maximum_CpuTime_Exceeded on a hard bilevel config) makes CAMPC return its
    # warmstart GUESS, whose first control is typically a SATURATED spin
    # (|omega| -> _OMEGA_MAX at ~half speed). Forwarding that whirls the robot in
    # place — the "robot does 360s" failure mode — so don't trust a failed solve.
    # SAFETY: with no collision-checked trajectory, never creep toward a nearby ped.
    solve_ok = isinstance(status, int) and status in (1, 2)
    if not solve_ok:
        clear = nearest_hum > _RECOVERY_CLEAR_DIST
        if clear:
            # Open space: align to the path lookahead (what path_foll tracks) and
            # creep forward once well-aligned, so a run of failures walks the robot
            # out of the hard config toward the goal (changing the geometry so the
            # next solve converges) instead of spinning.
            rerr = _drive_heading_err(global_plan, robot_pose, goal_xy)
            rec_v = 0.25 if abs(rerr) < _ALIGN_EXIT else 0.0
            rec_omega = float(np.clip(rerr / _ALIGN_TC, -_OMEGA_MAX, _OMEGA_MAX))
        else:
            # Near a ped (e.g. at a crossing): the path lookahead heading swings as
            # navfn re-routes around the moving ped, so chasing it spins the robot in
            # place. Hold position (v=0) and turn GENTLY toward the STABLE goal
            # bearing (capped at _REC_OMEGA_CAP) so the robot waits for a gap facing
            # roughly the goal, ready to drive once a solve converges, rather than
            # whirling. Turning in place never reduces clearance.
            rerr = (math.atan2(goal_xy[1] - ry, goal_xy[0] - rx) - theta + math.pi) % (2 * math.pi) - math.pi
            rec_v = 0.0
            rec_omega = float(np.clip(rerr / _ALIGN_TC, -_REC_OMEGA_CAP, _REC_OMEGA_CAP))
        _dbg(f"solve NOT converged (status={status}) -> recovery [v={rec_v}, omega={rec_omega:.3f}] "
             f"(nearest_hum={nearest_hum:.2f} clear={clear} rerr={rerr:.2f}; discarded raw omega={omega:.3f})")
        # A sustained failed-solve hold next to a ped is itself a freeze; let the
        # go-around escalate out of it instead of waiting forever for a solve.
        return _unfreeze(rec_v, rec_omega, robot_pose, humans, goal_xy, global_plan)

    # Reactive safety brake (prediction-independent final guard): even on a converged
    # solve the MPC may have planned through a real pedestrian its internal model
    # mispredicted, so cap forward speed using measured pedestrian proximity/closing.
    # omega is preserved so the robot can still steer away while slowing/stopped.
    v = _safety_brake(v, robot_pose, humans)
    # Deadlock-breaking go-around: if a ped camps on the path and the (braked) MPC
    # command stays frozen, escalate to a slow lateral pass; otherwise pass through.
    return _unfreeze(v, omega, robot_pose, humans, goal_xy, global_plan)


def on_reset(episode_id: str, initial_state: dict | None) -> None:
    # Rebuild the MPC for the next episode (re-fixes human count, resets warmstart).
    global _policy, _aligning, _last_ref_key
    global _frozen_steps, _goaround_active
    _policy = None
    _aligning = False
    _last_ref_key = None
    _frozen_steps = 0
    _goaround_active = False


if __name__ == "__main__":
    manifest = load_manifest(pathlib.Path(__file__).parent / "planner.yaml")
    main_loop(step, manifest=manifest, on_reset=on_reset)
