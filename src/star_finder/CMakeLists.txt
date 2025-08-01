add_executable(CPUStarFinder CPUStarFinder.c)

target_link_libraries(CPUStarFinder cfitsio
                                    m)

if (CUDAToolkit_FOUND AND NOT CPU_ONLY)
    add_cuda_hip_executable(cudaStarFinder cudaStarFinder.cu)

    target_link_libraries(cudaStarFinder    cfitsio
                                            m)
endif()