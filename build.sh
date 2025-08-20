#!/usr/bin/env bash
# Simple Linux build script for gemv_quantv1.cpp using OpenCL
# Usage: ./build.sh [output_name]
set -euo pipefail

CXX=${CXX:-g++}
SRC=gemv_quantv1.cpp
OUT=${1:-gemv_quantv1}
CXXFLAGS_DEFAULT="-O3 -std=c++17 -DNDEBUG"
CXXFLAGS=${CXXFLAGS:-$CXXFLAGS_DEFAULT}

# Gather OpenCL flags via pkg-config when available; otherwise fall back to -lOpenCL
CL_CFLAGS=""
CL_LDFLAGS=""
if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists OpenCL; then
  CL_CFLAGS="$(pkg-config --cflags OpenCL)"
  CL_LDFLAGS="$(pkg-config --libs OpenCL)"
else
  # Fallback: common system defaults
  CL_LDFLAGS="-lOpenCL"
fi

# Optional manual overrides similar to the Windows .bat
# e.g. OPENCL_HEADERS=/usr/include OPENCL_LIB=/usr/lib/x86_64-linux-gnu ./build.sh
if [[ -n "${OPENCL_HEADERS:-}" ]]; then
  CL_CFLAGS+=" -I${OPENCL_HEADERS}"
fi
if [[ -n "${OPENCL_LIB:-}" ]]; then
  CL_LDFLAGS+=" -L${OPENCL_LIB}"
fi

set -x
"${CXX}" ${CXXFLAGS} ${CL_CFLAGS} -o "${OUT}" "${SRC}" ${CL_LDFLAGS}
set +x

echo "Build finished: ${OUT}"
