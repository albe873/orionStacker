add_cuda_hip_library(starFinder device_starFinder.cu)

target_sources(starFinder PRIVATE   host_starFinder.cpp
                                    descriptor_builder.cpp
                                    host_warp.cpp)

target_include_directories(starFinder PUBLIC include)
target_link_libraries(starFinder PRIVATE common ${OpenCV_LIBS})

if(OpenMP_CXX_FOUND)
    target_link_libraries(starFinder PRIVATE OpenMP::OpenMP_CXX)
endif()
