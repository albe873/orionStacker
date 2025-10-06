# Debayer
if (CUDAToolkit_FOUND AND NOT CPU_ONLY)
    add_cuda_hip_executable(cudaDebayerTest cudaDebayerTest.cu)
    target_link_libraries(cudaDebayerTest   common debayer cfitsio m)
endif()

# Stacker
add_executable(CPUStackerAlfaSigma CPUStackerAlfaSigma.c)
target_link_libraries(CPUStackerAlfaSigma   common stacker cfitsio m)

if (CUDAToolkit_FOUND AND NOT CPU_ONLY)
    add_cuda_hip_executable(cudaStackerAlfaSigma cudaStackerAlfaSigma.cu)
    target_link_libraries(cudaStackerAlfaSigma  common stacker cfitsio m)
endif()

# starFinder
add_executable(CPUStarFinder CPUStarFinder.c)
target_link_libraries(CPUStarFinder   common starFinder cfitsio m)

if (CUDAToolkit_FOUND AND NOT CPU_ONLY)
    add_cuda_hip_executable(cudaStarFinder cudaStarFinder.cu)
    target_link_libraries(cudaStarFinder  common starFinder cfitsio m)
endif()

# features

# Find OpenCV with xfeatures2d for SURF support
#find_package( OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui features2d xfeatures2d )
#include_directories( ${OpenCV_INCLUDE_DIRS} )

#add_executable(features features.cpp)
#target_link_libraries(features   ${OpenCV_LIBS})
