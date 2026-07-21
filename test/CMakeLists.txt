include_directories( ${OpenCV_INCLUDE_DIRS} )


# ===== Debayer =====
if (CUDAToolkit_FOUND AND NOT CPU_ONLY)
    add_cuda_hip_executable(cudaDebayerTest cudaDebayerTest.cu)
    target_link_libraries(cudaDebayerTest   common debayer cfitsio m)
endif()

# ===== calibration =====
if (CUDAToolkit_FOUND AND NOT CPU_ONLY)
    add_cuda_hip_executable(cudaCalibration cudaCalibration.cu)
    target_link_libraries(cudaCalibration   common calibration cfitsio m)
endif()

# ===== Stacker =====
if (CUDAToolkit_FOUND AND NOT CPU_ONLY)
    add_cuda_hip_executable(cudaStackerAlfaSigma cudaStackerAlfaSigma.cu)
    target_link_libraries(cudaStackerAlfaSigma  common stacker cfitsio m)
endif()

# ===== starFinder =====
if (CUDAToolkit_FOUND AND NOT CPU_ONLY)
    add_cuda_hip_executable(cudaStarFinder cudaStarFinder.cu)
    target_link_libraries(cudaStarFinder  common starFinder cfitsio m ${OpenCV_LIBS})
endif()

# ===== aligner =====
if (CUDAToolkit_FOUND AND NOT CPU_ONLY)
    add_cuda_hip_executable(cudaAligner cudaAligner.cu)
    target_link_libraries(cudaAligner  common starFinder debayer cfitsio m ${OpenCV_LIBS})
endif()

# ===== features =====
add_executable(features features.cpp)
target_link_libraries(features   ${OpenCV_LIBS})

# ===== testAll =====
if (CUDAToolkit_FOUND AND NOT CPU_ONLY)
    add_cuda_hip_executable(testAll testAll.cu)
    target_link_libraries(testAll common calibration debayer starFinder stacker cfitsio m)
endif()
