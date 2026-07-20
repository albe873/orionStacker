add_cuda_hip_library(stacker device_alfa_sigma.cu)
target_sources(stacker PRIVATE host_alfa_sigma.cpp)

target_include_directories(stacker PUBLIC include)
target_link_libraries(stacker PRIVATE common)

if(OpenMP_CXX_FOUND)
    target_link_libraries(stacker PRIVATE OpenMP::OpenMP_CXX)
endif()
