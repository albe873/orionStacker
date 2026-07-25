#!/usr/bin/env bash
# Build OpenCV statically from the local submodule source at third_party/opencv
# Tracks the opencv/opencv 4.x branch.
# Usage: ./third_party/build_opencv_static.sh [install_prefix]
#
# Default install prefix: /workspaces/cudastacker/third_party/opencv_install
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OPENCV_SOURCE_DIR="${SCRIPT_DIR}/opencv"
INSTALL_PREFIX="${1:-${SCRIPT_DIR}/opencv_install}"

if [ ! -d "$OPENCV_SOURCE_DIR" ]; then
    echo "ERROR: OpenCV submodule source not found at $OPENCV_SOURCE_DIR"
    echo "Please run: git submodule update --init --recursive"
    exit 1
fi

BUILD_DIR="${SCRIPT_DIR}/opencv_build"
mkdir -p "$BUILD_DIR"

echo "=== Building OpenCV statically ==="
echo "Source:       $OPENCV_SOURCE_DIR"
echo "Build dir:    $BUILD_DIR"
echo "Install dir:  $INSTALL_PREFIX"
echo ""

# ---------------------------------------------------------------
# Step 1: Build ADE (OpenCV graph dependency) from source
# ---------------------------------------------------------------
ADE_SRC_DIR="${SCRIPT_DIR}/ade"
ADE_BUILD_DIR="${BUILD_DIR}/ade_build"
ADE_INSTALL_DIR="${BUILD_DIR}/ade_install"

if [ ! -d "$ADE_SRC_DIR" ]; then
    echo "=== Cloning ADE framework ==="
    git clone --depth=1 https://github.com/opencv/ade.git "$ADE_SRC_DIR"
fi

echo "=== Building ADE statically ==="
mkdir -p "$ADE_BUILD_DIR"
cd "$ADE_BUILD_DIR"
cmake "$ADE_SRC_DIR" \
    -DCMAKE_INSTALL_PREFIX="$ADE_INSTALL_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF
cmake --build . --parallel "$(nproc)"
cmake --install .

# Create a cmake config for ADE so OpenCV can find it.
ADE_CMAKE_DIR="${ADE_INSTALL_DIR}/lib/cmake/ade"
mkdir -p "$ADE_CMAKE_DIR"
cat > "${ADE_CMAKE_DIR}/ade-config.cmake" << EOF
add_library(ade STATIC IMPORTED)
set_target_properties(ade PROPERTIES
    IMPORTED_LOCATION "${ADE_INSTALL_DIR}/lib/libade.a"
    INTERFACE_INCLUDE_DIRECTORIES "${ADE_SRC_DIR}/source/include"
)
set(ade_VERSION 0.1.0)
set(ade_VERSION_MAJOR 0)
set(ade_VERSION_MINOR 1)
set(ade_VERSION_PATCH 0)
EOF
cp "${ADE_CMAKE_DIR}/ade-config.cmake" "${ADE_CMAKE_DIR}/adeConfig.cmake"

echo "ADE built and installed to $ADE_INSTALL_DIR"

# ---------------------------------------------------------------
# Step 2: Build OpenCV with the local ADE
# ---------------------------------------------------------------
echo ""
echo "=== Building OpenCV statically (with local ADE) ==="

cd "$BUILD_DIR"

cmake "$OPENCV_SOURCE_DIR" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_LIST=core,imgproc,imgcodecs,highgui,features2d,calib3d,flann \
    -DBUILD_opencv_world=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_DOCS=OFF \
    -DWITH_GTK=ON \
    -DWITH_GTK_2_X=OFF \
    -DWITH_V4L=OFF \
    -DWITH_FFMPEG=OFF \
    -DWITH_GSTREAMER=OFF \
    -DWITH_1394=OFF \
    -DWITH_OPENCL=OFF \
    -DWITH_OPENEXR=OFF \
    -DWITH_CAROTENE=OFF \
    -DWITH_ITT=OFF \
    -DWITH_PROTOBUF=OFF \
    -DWITH_QUIRC=OFF \
    -DWITH_TBB=OFF \
    -DWITH_EIGEN=OFF \
    -DWITH_WEBP=ON \
    -DWITH_JPEG=ON \
    -DWITH_PNG=ON \
    -DWITH_TIFF=ON \
    -DWITH_JASPER=OFF \
    -DWITH_OPENJPEG=ON \
    -DENABLE_PRECOMPILED_HEADERS=OFF \
    -DCPU_BASELINE=DETECT \
    -DBUILD_ZLIB=ON \
    -DBUILD_PNG=ON \
    -DBUILD_JPEG=ON \
    -DBUILD_TIFF=ON \
    -DBUILD_WEBP=ON \
    -DBUILD_OPENJPEG=ON \
    -DOPENCV_DOWNLOAD_ADE=OFF \
    -Dade_DIR="${ADE_INSTALL_DIR}/lib/cmake/ade"

echo ""
echo "=== Building (using all available cores) ==="
cmake --build . --parallel "$(nproc)"

echo ""
echo "=== Installing to $INSTALL_PREFIX ==="
cmake --install .

# If libade.a wasn't placed in the 3rdparty directory by OpenCV, copy it there.
OPENCV_3RDPARTY_DIR="${INSTALL_PREFIX}/lib/opencv4/3rdparty"
if [ ! -f "${OPENCV_3RDPARTY_DIR}/libade.a" ]; then
    mkdir -p "$OPENCV_3RDPARTY_DIR"
    cp "${ADE_INSTALL_DIR}/lib/libade.a" "${OPENCV_3RDPARTY_DIR}/libade.a"
    echo "Copied libade.a to ${OPENCV_3RDPARTY_DIR}/libade.a"
fi

echo ""
echo "=== Done! ==="
echo "OpenCV static libraries installed to: $INSTALL_PREFIX"
echo ""
echo "To use in CMake, set: -DOpenCV_DIR=$INSTALL_PREFIX/lib/cmake/opencv4"
