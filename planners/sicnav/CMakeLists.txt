cmake_minimum_required(VERSION 3.12)
project(arena_planners_sicnav NONE)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_venv REQUIRED)
find_package(ament_cmake_venv_uv REQUIRED)

# Build the per-planner venv from pyproject.toml (CasADi/IPOPT + SICNav + bridge).
uv_venv(PROJECTFILE pyproject.toml)

# Python-RVO2 (rvo2) is required by SICNav's ORCA-KKT callback. It has no
# pyproject, imports Cython in setup.py, and builds a bundled C++ lib via CMake,
# so it can't go through uv's isolated resolver. Install it into the venv with
# build isolation disabled, against the venv's cython/setuptools/wheel (system
# cmake + g++ build the C++). Runs after the venv stamp from uv_venv.
set(_rvo2_src "${CMAKE_CURRENT_SOURCE_DIR}/deps/Python-RVO2")
set(_rvo2_build "${CMAKE_CURRENT_BINARY_DIR}/rvo2_src")
set(_rvo2_stamp "${CMAKE_CURRENT_BINARY_DIR}/rvo2.installed")

add_custom_command(
  OUTPUT "${_rvo2_stamp}"
  DEPENDS "${venv_dir}" "${_rvo2_src}/setup.py"
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${_rvo2_build}"
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${_rvo2_src}" "${_rvo2_build}"
  COMMAND sh -c "export PYTHONPATH='' && cd '${_rvo2_build}' && CC=gcc CXX=g++ uv pip install --python '${venv_python_bin}' --no-build-isolation ."
  COMMAND ${CMAKE_COMMAND} -E touch "${_rvo2_stamp}"
  COMMENT "Installing Python-RVO2 (rvo2) into the SICNav venv"
  VERBATIM
)
add_custom_target(RVO2_INSTALL ALL DEPENDS "${_rvo2_stamp}" "${target_venv}")

# The upstream (third-party) SICNav campc.py builds its IPOPT solver without an
# ipopt.max_cpu_time bound, so a hard bilevel solve can run for seconds and the
# robot drives blind. We don't own that fork, so re-apply the one-line solver-time
# bound to the venv copy on every (re)build. Idempotent; see scripts/.
set(_campc_patch "${CMAKE_CURRENT_SOURCE_DIR}/scripts/patch_campc_solver.py")
set(_campc_stamp "${CMAKE_CURRENT_BINARY_DIR}/campc_patched.installed")
add_custom_command(
  OUTPUT "${_campc_stamp}"
  DEPENDS "${venv_dir}" "${_campc_patch}"
  COMMAND "${venv_python_bin}" "${_campc_patch}"
  COMMAND ${CMAKE_COMMAND} -E touch "${_campc_stamp}"
  COMMENT "Patching SICNav campc.py with ipopt.max_cpu_time bound (solver safety)"
  VERBATIM
)
add_custom_target(CAMPC_PATCH ALL DEPENDS "${_campc_stamp}" "${target_venv}")

install(
  FILES
    planner.py
    planner.yaml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
