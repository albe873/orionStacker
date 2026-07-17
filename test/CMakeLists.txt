include_directories( ${OpenCV_INCLUDE_DIRS} )


# Debayer
if (CUDAToolkit_FOUND AND NOT CPU_ONLY)
    add_cuda_hip_executable(cudaDebayerTest cudaDebayerTest.cu)
    target_link_libraries(cudaDebayerTest   common debayer cfitsio m)
endif()

# calibration
if (CUDAToolkit_FOUND AND NOT CPU_ONLY)
    add_cuda_hip_executable(cudaCalibration cudaCalibration.cu)
    target_link_libraries(cudaCalibration   common calibration cfitsio m)
endif()

# Stacker
add_executable(CPUStackerAlfaSigma CPUStackerAlfaSigma.cpp)
target_link_libraries(CPUStackerAlfaSigma   common stacker cfitsio m)

if (CUDAToolkit_FOUND AND NOT CPU_ONLY)
    add_cuda_hip_executable(cudaStackerAlfaSigma cudaStackerAlfaSigma.cu)
    target_link_libraries(cudaStackerAlfaSigma  common stacker cfitsio m)
endif()

# starFinder
#add_executable(CPUStarFinder CPUStarFinder.cpp)
#target_link_libraries(CPUStarFinder   common starFinder cfitsio m)

if (CUDAToolkit_FOUND AND NOT CPU_ONLY)
    add_cuda_hip_executable(cudaStarFinder cudaStarFinder.cu)
    target_link_libraries(cudaStarFinder  common starFinder cfitsio m ${OpenCV_LIBS})
endif()

# starFinder - Otsu Centralized Threshold Test
#if (CUDAToolkit_FOUND AND NOT CPU_ONLY)
#    add_cuda_hip_executable(cudaStarFinder2 cudaStarFinder2.cu)
#    target_link_libraries(cudaStarFinder2  common starFinder cfitsio m ${OpenCV_LIBS})
#endif()

# aligner
if (CUDAToolkit_FOUND AND NOT CPU_ONLY)
    add_cuda_hip_executable(cudaAligner cudaAligner.cu)
    target_link_libraries(cudaAligner  common starFinder debayer cfitsio m ${OpenCV_LIBS})
endif()

# features
add_executable(features features.cpp)
target_link_libraries(features   ${OpenCV_LIBS})
