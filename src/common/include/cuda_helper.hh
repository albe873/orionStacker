#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <cstdio>

#ifdef __HIPCC__
    #include <hip/hip_runtime.h>
    typedef hipError_t cudaError_t;
    #define cudaSuccess hipSuccess
    #define cudaError_t hipError_t
    #define cudaGetErrorString hipGetErrorString
    #define cudaMalloc hipMalloc
    #define cudaMemcpy hipMemcpy
    #define cudaMemset hipMemset
    #define cudaFree hipFree
    #define cudaDeviceSynchronize hipDeviceSynchronize
    #define cudaMemcpyHostToDevice hipMemcpyHostToDevice
    #define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#else
    #include <cuda_runtime.h>
#endif




#if defined(__HIP_PLATFORM_AMD__)
using PrefetchDeviceArg = int;
static inline PrefetchDeviceArg make_prefetch_device_arg(int dev) { return dev; }
#else
using PrefetchDeviceArg = cudaMemLocation;
static inline PrefetchDeviceArg make_prefetch_device_arg(int dev) {
    PrefetchDeviceArg loc;
    loc.id = dev;
    loc.type = cudaMemLocationTypeDevice;
    return loc;
}
#endif

#define CHECK(err) do { cuda_check((err), __FILE__, __LINE__); } while(false)
inline void cuda_check(cudaError_t error_code, const char *file, int line) {
    if (error_code != cudaSuccess) {
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
        fflush(stderr);
        exit(error_code);
    }
}

#endif // CUDA_HELPER_H