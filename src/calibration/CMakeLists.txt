add_cuda_hip_library(calibration device_calibration.cu)
target_sources(calibration PRIVATE host_calibration.cpp)

target_include_directories(calibration PUBLIC include)
target_link_libraries(calibration PRIVATE common)

if(OpenMP_CXX_FOUND)
    target_link_libraries(calibration PRIVATE OpenMP::OpenMP_CXX)
endif()