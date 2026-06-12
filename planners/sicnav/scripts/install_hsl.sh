#!/usr/bin/env bash
#
# install_hsl.sh — build Coin-HSL (MA57) and wire it into the SICNav planner venv
# so IPOPT/CasADi use the MA57 linear solver instead of the slower MUMPS default.
#
# WHY: SICNav's MPC (campc.py) probes for MA57 at startup and uses it if the HSL
# library can be loaded. MA57 brings multi-human solves back to real time
# (~8 s/step at 5 humans on MUMPS -> well under the control period on MA57).
#
# LICENSING: Coin-HSL is distributed under a per-user academic licence and MUST
# NOT be committed to this repo. You supply your own source tarball; this script
# only builds and installs it locally. Get it free for academics at
# https://licences.stfc.ac.uk/product/coin-hsl (download the full "Coin-HSL",
# which includes MA57 — not the smaller "Coin-HSL Archive").
#
# RUN THIS INSIDE THE ARENA CONTAINER, e.g.:
#   ./arena exec   # (or: docker exec -it <arena-container> bash)
#   bash src/Arena/arena_planners/planners/sicnav/scripts/install_hsl.sh \
#        /opt/arena_ws/build/hsl_build/coinhsl-2024.05.15.tar.gz
#
# If you omit the tarball path it searches a few common locations.
# Re-run this after any clean rebuild of the SICNav venv (a venv rebuild
# reinstalls casadi and removes the bundled libhsl.so).
#
set -euo pipefail

# ---- locate the Coin-HSL source tarball -------------------------------------
TARBALL="${1:-}"
if [ -z "$TARBALL" ]; then
    for c in \
        /opt/arena_ws/build/hsl_build/coinhsl-*.tar.gz \
        /opt/arena_ws/hsl/coinhsl-*.tar.gz \
        "$HOME"/coinhsl-*.tar.gz ; do
        if [ -f "$c" ]; then TARBALL="$c"; break; fi
    done
fi
if [ -z "$TARBALL" ] || [ ! -f "$TARBALL" ]; then
    echo "ERROR: Coin-HSL source tarball not found." >&2
    echo "Pass it explicitly: install_hsl.sh /path/to/coinhsl-YYYY.MM.DD.tar.gz" >&2
    exit 1
fi
echo ">> Using Coin-HSL source: $TARBALL"

# ---- locate the SICNav venv's casadi package (the IPOPT load dir) -----------
# libipopt.so dlopens "libhsl.so" and has RPATH=$ORIGIN, so a libhsl.so placed
# in the casadi package dir is found at runtime with no env/launch changes.
VENV_PY="$(find /opt/arena_ws/build -maxdepth 4 -path '*arena_planners_sicnav/venv/bin/python' 2>/dev/null | head -1)"
if [ -z "$VENV_PY" ]; then
    echo "ERROR: SICNav venv not found. Build the planner first: arena build arena_planners_sicnav" >&2
    exit 1
fi
CASADI_DIR="$("$VENV_PY" -c 'import casadi, os; print(os.path.dirname(casadi.__file__))')"
echo ">> SICNav venv python: $VENV_PY"
echo ">> casadi dir:         $CASADI_DIR"

# ---- build dependencies ------------------------------------------------------
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
export DEBIAN_FRONTEND=noninteractive
echo ">> Installing build deps (gfortran, meson, ninja, libmetis-dev, patchelf)..."
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq \
    gfortran meson ninja-build libmetis-dev patchelf >/dev/null

# ---- build Coin-HSL ----------------------------------------------------------
WORK=/opt/arena_ws/build/hsl_build
mkdir -p "$WORK"
cd "$WORK"
# Top-level dir matches the tarball name (coinhsl-YYYY.MM.DD.tar.gz -> coinhsl-YYYY.MM.DD).
# Derive it from the filename rather than `tar | head` (which SIGPIPEs under pipefail).
SRCDIR="$(basename "$TARBALL")"; SRCDIR="${SRCDIR%.tar.gz}"; SRCDIR="${SRCDIR%.tgz}"
rm -rf "$SRCDIR/builddir"
tar xzf "$TARBALL"
cd "$SRCDIR"
echo ">> Configuring (meson)..."
meson setup builddir --buildtype=release \
    -Dlibblas=blas -Dliblapack=lapack -Dlibmetis=metis -Dlibmetis_version=5
echo ">> Compiling..."
meson compile -C builddir
BUILT="$PWD/builddir/libcoinhsl.so"
[ -f "$BUILT" ] || { echo "ERROR: libcoinhsl.so was not produced" >&2; exit 1; }

# ---- install into the casadi dir, self-contained ----------------------------
# Bundle libmetis.so.5 next to libhsl.so and point RPATH at $ORIGIN so the lib
# keeps working even if the apt-installed libmetis is gone after a container
# rebuild. libblas/liblapack/libgfortran are base-image libs (always present).
echo ">> Installing libhsl.so into casadi dir (self-contained)..."
cp -f "$BUILT" "$CASADI_DIR/libcoinhsl.so"
# tail -1 reads all input (no early pipe close, so no SIGPIPE under pipefail)
METIS_SO="$(ldconfig -p | grep 'libmetis.so.5' | tail -1 | awk '{print $NF}')"
if [ -n "$METIS_SO" ] && [ -f "$METIS_SO" ]; then
    cp -fL "$METIS_SO" "$CASADI_DIR/libmetis.so.5"
fi
patchelf --set-rpath '$ORIGIN' "$CASADI_DIR/libcoinhsl.so"
ln -sf libcoinhsl.so "$CASADI_DIR/libhsl.so"

# ---- verify MA57 actually loads (same probe campc.py uses) ------------------
echo ">> Verifying MA57 is available to IPOPT..."
"$VENV_PY" - <<'PY'
import sys, casadi as cs
x = cs.MX.sym("x", 2)
prob = {"x": x, "f": cs.dot(x, x), "g": x[0] + x[1]}
try:
    s = cs.nlpsol("d", "ipopt", prob,
                  {"ipopt.linear_solver": "ma57", "ipopt.print_level": 0, "print_time": 0})
    s(x0=[1, 1], lbg=1, ubg=1)
    print(">>> SUCCESS: IPOPT loaded MA57. SICNav will use it automatically.")
except Exception as e:
    print(">>> FAILED: MA57 did not load:", repr(e)[:300]); sys.exit(1)
PY
echo ">> Done. SICNav (campc.py) auto-detects MA57 on startup; no further config needed."
