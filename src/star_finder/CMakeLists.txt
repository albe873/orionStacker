add_cuda_hip_library(starFinder device_starFinder.cu)

target_sources(starFinder PRIVATE host_starFinder.cpp)

target_include_directories(starFinder PUBLIC include)
target_link_libraries(starFinder  common)
