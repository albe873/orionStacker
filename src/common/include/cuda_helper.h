#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#if defined(__HIPCC__)
    typedef hipError_t cudaError_t;
    #define cudaSuccess hipSuccess
    #define cudaGetErrorString hipGetErrorString
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

#include <cstdio>

#define CHECK(err) do { cuda_check((err), __FILE__, __LINE__); } while(false)
inline void cuda_check(cudaError_t error_code, const char *file, int line) {
    if (error_code != cudaSuccess) {
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
        fflush(stderr);
        exit(error_code);
    }
}

#endif // CUDA_HELPER_H