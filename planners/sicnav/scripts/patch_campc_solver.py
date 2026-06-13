#!/usr/bin/env python3
"""Durably apply the SICNav solver-safety patch to the vendored CAMPC.

The original SICNav ``campc.py`` (installed into the planner venv from the
upstream git dep) builds its IPOPT solver with ``ipopt.max_iter`` but NO
``ipopt.max_cpu_time``. The bilevel ORCA-KKT solve then occasionally runs for
multiple seconds; while it does, Arena's bridge holds the last ``cmd_vel`` and
the robot drives BLIND into pedestrians. Bounding the IPOPT wall-clock makes the
solve return its best feasible point within the control budget instead.

Upstream is a third-party fork we don't own, so we can't fix it there. This
script re-applies the one-line option to the venv copy and is wired into
CMakeLists.txt so it runs automatically after every venv (re)build — the same
durability story as the RVO2 install and the HSL drop-in.

Idempotent: running it again on an already-patched file is a no-op. Run it with
the planner venv's python so it resolves the right ``sicnav`` install:

    <venv>/bin/python scripts/patch_campc_solver.py [max_cpu_time_seconds]
"""

from __future__ import annotations

import importlib.util
import sys

MARKER = "ipopt.max_cpu_time"
ANCHOR = "opti.solver('ipopt', opts)"
DEFAULT_MAX_CPU_TIME = 0.45


def main() -> int:
    max_cpu_time = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MAX_CPU_TIME

    spec = importlib.util.find_spec("sicnav.policy.campc")
    if spec is None or not spec.origin:
        print("[patch_campc] ERROR: sicnav.policy.campc not found in this python env", file=sys.stderr)
        return 1
    path = spec.origin

    with open(path) as fh:
        src = fh.read()

    if MARKER in src:
        print(f"[patch_campc] already patched ({MARKER} present): {path}")
        return 0

    lines = src.splitlines(keepends=True)
    out = []
    inserted = False
    for line in lines:
        if not inserted and ANCHOR in line:
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}# [Arena port patch] bound the IPOPT wall-clock to the control period so a\n")
            out.append(f"{indent}# hard bilevel solve can't spike to multiple seconds and make the robot\n")
            out.append(f"{indent}# drive blind. See scripts/patch_campc_solver.py. Re-applied on each build.\n")
            out.append(f'{indent}opts["{MARKER}"] = {max_cpu_time}\n')
            inserted = True
        out.append(line)

    if not inserted:
        print(f"[patch_campc] ERROR: anchor not found ({ANCHOR!r}); upstream layout changed: {path}", file=sys.stderr)
        return 1

    with open(path, "w") as fh:
        fh.write("".join(out))
    print(f"[patch_campc] applied {MARKER}={max_cpu_time} to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
