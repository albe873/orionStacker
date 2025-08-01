add_executable(CPUStackerAlfaSigma CPUStackerAlfaSigma.c)

target_link_libraries(CPUStackerAlfaSigma   cfitsio
                                            m)

if (CUDAToolkit_FOUND AND NOT CPU_ONLY)
    add_cuda_hip_executable(cudaStackerAlfaSigma cudaStackerAlfaSigma.cu)
    target_link_libraries(cudaStackerAlfaSigma  cfitsio
                                                m)
endif()
