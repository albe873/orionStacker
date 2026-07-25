# Build OpenCV statically from the local submodule source at opencv/
#
# This file provides an ExternalProject-based build so that CMake can
# automatically configure and build OpenCV if it hasn't been pre-built.
#
# Usage (from root CMakeLists.txt):
#   add_subdirectory(third_party)
#   set(OpenCV_DIR "${CMAKE_BINARY_DIR}/opencv_install/lib/cmake/opencv4" CACHE PATH "")
#   find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui features2d calib3d flann gapi)

include(ExternalProject)

set(OPENCV_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/opencv")
set(OPENCV_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/opencv_install")

# ---------------------------------------------------------------
# Build ADE (OpenCV graph dependency) from source
# ---------------------------------------------------------------
set(ADE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ade")
set(ADE_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/ade_install")

if(EXISTS "${ADE_SOURCE_DIR}/CMakeLists.txt")
    ExternalProject_Add(ade_static
        SOURCE_DIR          "${ADE_SOURCE_DIR}"
        PREFIX              "${CMAKE_CURRENT_BINARY_DIR}/ade_ep"
        INSTALL_DIR         "${ADE_INSTALL_DIR}"
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${ADE_INSTALL_DIR}
            -DCMAKE_BUILD_TYPE=Release
            -DBUILD_SHARED_LIBS=OFF
        BUILD_BYPRODUCTS
            "${ADE_INSTALL_DIR}/lib/libade.a"
        EXCLUDE_FROM_ALL     TRUE
    )

    # ADE doesn't ship a cmake config file — create one after build.
    set(ADE_CMAKE_DIR "${ADE_INSTALL_DIR}/lib/cmake/ade")
    file(MAKE_DIRECTORY "${ADE_CMAKE_DIR}")
    file(WRITE "${ADE_CMAKE_DIR}/ade-config.cmake"
"# ade-config.cmake - generated for static build
add_library(ade STATIC IMPORTED)
set_target_properties(ade PROPERTIES
    IMPORTED_LOCATION \"${ADE_INSTALL_DIR}/lib/libade.a\"
    INTERFACE_INCLUDE_DIRECTORIES \"${ADE_SOURCE_DIR}/source/include\"
)
set(ade_VERSION 0.1.0)
set(ade_VERSION_MAJOR 0)
set(ade_VERSION_MINOR 1)
set(ade_VERSION_PATCH 0)
")
    file(WRITE "${ADE_CMAKE_DIR}/adeConfig.cmake"
"# adeConfig.cmake - generated for static build
add_library(ade STATIC IMPORTED)
set_target_properties(ade PROPERTIES
    IMPORTED_LOCATION \"${ADE_INSTALL_DIR}/lib/libade.a\"
    INTERFACE_INCLUDE_DIRECTORIES \"${ADE_SOURCE_DIR}/source/include\"
)
set(ade_VERSION 0.1.0)
set(ade_VERSION_MAJOR 0)
set(ade_VERSION_MINOR 1)
set(ade_VERSION_PATCH 0)
")
endif()

# ---------------------------------------------------------------
# Build OpenCV statically with local ADE
# ---------------------------------------------------------------
# Only define the external project if the submodule source exists.
if(EXISTS "${OPENCV_SOURCE_DIR}/CMakeLists.txt")
    ExternalProject_Add(opencv_static
        SOURCE_DIR          "${OPENCV_SOURCE_DIR}"
        PREFIX              "${CMAKE_CURRENT_BINARY_DIR}/opencv_ep"
        INSTALL_DIR         "${OPENCV_INSTALL_DIR}"
        DEPENDS             ade_static
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${OPENCV_INSTALL_DIR}
            -DCMAKE_BUILD_TYPE=Release
            -DBUILD_SHARED_LIBS=OFF
            -DBUILD_LIST=core,imgproc,imgcodecs,highgui,features2d,calib3d,flann
            -DBUILD_opencv_world=OFF
            -DBUILD_EXAMPLES=OFF
            -DBUILD_TESTS=OFF
            -DBUILD_PERF_TESTS=OFF
            -DBUILD_DOCS=OFF
            -DWITH_GTK=ON
            -DWITH_GTK_2_X=OFF
            -DWITH_V4L=OFF
            -DWITH_FFMPEG=OFF
            -DWITH_GSTREAMER=OFF
            -DWITH_1394=OFF
            -DWITH_OPENCL=OFF
            -DWITH_OPENEXR=OFF
            -DWITH_CAROTENE=OFF
            -DWITH_ITT=OFF
            -DWITH_PROTOBUF=OFF
            -DWITH_QUIRC=OFF
            -DWITH_TBB=OFF
            -DWITH_EIGEN=OFF
            -DWITH_WEBP=ON
            -DWITH_JPEG=ON
            -DWITH_PNG=ON
            -DWITH_TIFF=ON
            -DWITH_JASPER=OFF
            -DWITH_OPENJPEG=ON
            -DENABLE_PRECOMPILED_HEADERS=OFF
            -DCPU_BASELINE=DETECT
            -DBUILD_ZLIB=ON
            -DBUILD_PNG=ON
            -DBUILD_JPEG=ON
            -DBUILD_TIFF=ON
            -DBUILD_WEBP=ON
            -DBUILD_OPENJPEG=ON
            -DOPENCV_DOWNLOAD_ADE=OFF
            -Dade_DIR=${ADE_INSTALL_DIR}/lib/cmake/ade
        BUILD_ALWAYS         FALSE
        BUILD_BYPRODUCTS
            "${OPENCV_INSTALL_DIR}/lib/cmake/opencv4/OpenCVConfig.cmake"
        EXCLUDE_FROM_ALL     TRUE
    )

    # Expose the install directory so parent can use it with find_package.
    set(OpenCV_DIR "${OPENCV_INSTALL_DIR}/lib/cmake/opencv4" PARENT_SCOPE)
    message(STATUS "OpenCV static build configured. Install dir: ${OPENCV_INSTALL_DIR}")
else()
    message(WARNING
        "OpenCV submodule source not found at ${OPENCV_SOURCE_DIR}.\n"
        "Run:  git submodule update --init --recursive\n"
        "Static OpenCV will NOT be auto-built."
    )
endif()
