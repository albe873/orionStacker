add_cuda_hip_library(debayer device_debayer.cu)
target_sources(debayer PRIVATE host_debayer.cpp)


target_include_directories(debayer PUBLIC include)
target_link_libraries(debayer common)
