if (CUDAToolkit_FOUND AND NOT CPU_ONLY)
    add_cuda_hip_executable(cudaDebayerTest cudaDebayerTest.cu)
    target_link_libraries(cudaDebayerTest   common
                                            debayer
                                            cfitsio
                                            m)
endif()