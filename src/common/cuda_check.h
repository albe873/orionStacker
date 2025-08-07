#ifndef CUDA_CHECK_H
#define CUDA_CHECK_H

#if defined(__HIPCC__)
    typedef hipError_t cudaError_t;
    #define cudaSuccess hipSuccess
    #define cudaGetErrorString hipGetErrorString
#endif

#define CHECK(err) do { cuda_check((err), __FILE__, __LINE__); } while(false)
inline void cuda_check(cudaError_t error_code, const char *file, int line) {
    if (error_code != cudaSuccess) {
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
        fflush(stderr);
        exit(error_code);
    }
}

#endif // CUDA_CHECK_H