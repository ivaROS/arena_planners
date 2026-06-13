#!/usr/bin/env python3
"""Durably restore SICNav's cruise speed by weighting the velocity state in Q.

Symptom: the robot crawls at ~0.35 m/s on open ground even though pref_speed is
0.9 and there is no velocity penalty and plenty of runway. Cause: SICNav's MPC
cost weights ONLY the x/y position (``Q = diag([1, 1, 0, 0, ...])`` in
``mpc_env.MPCEnv`` — index 3, the robot's velocity state ``v_prev``, has weight
0). Position tracking alone should still drive ~pref_speed, but the bilevel
ORCA-KKT problem is non-convex and the solver uses ``acceptable_*`` early-stop
tolerances together with a shift-forward (``bring_fwd``) warmstart. Each solve
therefore re-seeds from the previous SLOW trajectory and converges back to that
slow local minimum, so the robot never accelerates even when the acceleration
limit and the 1.8 m terminal target both reward it.

Fix: give the velocity state (index 3) a nonzero weight in the stage cost ``Q``.
The reference trajectory ALREADY carries pref_speed in that row (campc.generate_traj
builds ``ref_X`` row 3 = [|v0|, ref_v...] with ref_v = pref_speed), so weighting it
adds a direct cost gradient that pulls v -> pref_speed from the first iteration and
lets the solver climb out of the slow warmstart. The weight is kept well below the
ORCA collision-slack penalties, so hard avoidance still dominates speed.

Upstream is a third-party fork we don't own, so we can't fix it there. This script
re-applies the one-line change to the venv copy and is wired into CMakeLists.txt so
it runs automatically after every venv (re)build — same durability story as
patch_campc_solver.py.

Idempotent: running it again on an already-patched file is a no-op. Run it with the
planner venv's python so it resolves the right ``sicnav`` install:

    <venv>/bin/python scripts/patch_mpc_speed.py [velocity_weight]
"""

from __future__ import annotations

import importlib.util
import sys

MARKER = "[Arena port patch] cruise-speed"
# The exact upstream line that builds the stage cost weight matrix. Index 3 of the
# diagonal (the robot's v_prev state) is currently 0 -> velocity is not tracked.
ANCHOR = "self.Q = cs.sparsify(np.diag(np.hstack([np.ones(2), np.zeros(self.nx_r-2+self.np_g+self.nx_hum*self.num_hums)])))"
DEFAULT_VELOCITY_WEIGHT = 2.0


def main() -> int:
    w_v = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_VELOCITY_WEIGHT

    spec = importlib.util.find_spec("sicnav.utils.mpc_utils.mpc_env")
    if spec is None or not spec.origin:
        print("[patch_mpc_speed] ERROR: sicnav.utils.mpc_utils.mpc_env not found", file=sys.stderr)
        return 1
    path = spec.origin

    with open(path) as fh:
        src = fh.read()

    if MARKER in src:
        print(f"[patch_mpc_speed] already patched ({MARKER} present): {path}")
        return 0

    if ANCHOR not in src:
        print(f"[patch_mpc_speed] ERROR: anchor not found; upstream layout changed: {path}", file=sys.stderr)
        return 1

    lines = src.splitlines(keepends=True)
    out = []
    for line in lines:
        if ANCHOR in line:
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}# {MARKER}: weight the robot velocity state (Q index 3) so the\n")
            out.append(f"{indent}# MPC tracks the pref_speed already present in the reference and the\n")
            out.append(f"{indent}# non-convex solve doesn't settle into a slow warmstart local min.\n")
            out.append(f"{indent}# See scripts/patch_mpc_speed.py. Re-applied on each build.\n")
            out.append(
                f"{indent}self.Q = cs.sparsify(np.diag(np.hstack(["
                f"np.ones(2), np.array([0.0, {w_v}]), "
                f"np.zeros(self.np_g+self.nx_hum*self.num_hums)])))\n"
            )
        else:
            out.append(line)

    with open(path, "w") as fh:
        fh.write("".join(out))
    print(f"[patch_mpc_speed] applied velocity_weight={w_v} to Q in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
