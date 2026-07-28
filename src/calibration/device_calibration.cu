#include "cuda_helper.hh"
#include "calibration.hh"

#include <algorithm>

__host__ __device__ inline uint16_t clamp_u16_from_f32(float x) {
    if (x <= 0.0f) return 0;
    if (x >= 65535.0f) return 65535;
    return (u_int16_t)(x + 0.5f);
}


__global__ void kernel_mean(const uint16_t* __restrict__ images,
                                  float* __restrict__ mean,
                                  uint64_t width, uint64_t height, int n_img) {
    uint64_t idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t idx2 = idx1 + 1;
    uint64_t npixels = width * height;
    if (idx2 >= npixels)
        return;

    double acc1 = 0.0, acc2 = 0.0;
    for (int i = 0; i < n_img; i++) {
        acc1 += (float) images[i * npixels + idx1];
        acc2 += (float) images[i * npixels + idx2];   
    }
    mean[idx1] = acc1 / n_img;
    mean[idx2] = acc2 / n_img;
    
}

__global__ void kernel_mean_subtract(const uint16_t* __restrict__ images,
                                     const float* __restrict__ bias,
                                     float* __restrict__ result,
                                     uint64_t width, uint64_t height, int count) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t npixels = width * height;

    if (idx >= npixels)
        return;

    // Per ogni pixel, sottrai il master bias e poi calcola la media escludendo i valori <= 0
    double acc = 0.0;
    for (int i = 0; i < count; i++) {
        float val = (float)images[i * npixels + idx] - bias[idx];
        acc += val;
    }
    result[idx] = acc / count;
}

void masterBias_gpu(const uint16_t* __restrict__ bias_all,
                    float* __restrict__ master_bias,
                    uint64_t width, uint64_t height, int bias_count) {
    uint64_t npixels = width * height;
    
    dim3 block_size(512);
    dim3 grid_size((npixels + block_size.x - 1)/block_size.x);
    kernel_mean<<<grid_size, block_size>>>(bias_all, master_bias, width, height, bias_count);
    CHECK(cudaDeviceSynchronize());
}

void masterDark_gpu(const uint16_t* __restrict__ dark_all,
                    const float* __restrict__ master_bias,
                    float* __restrict__ master_dark,
                    uint64_t width, uint64_t height, int dark_count) {
    uint64_t npixels = width * height;

    dim3 block_size(512);
    dim3 grid_size((npixels + block_size.x - 1)/block_size.x);
    kernel_mean<<<grid_size, block_size>>>(dark_all, master_dark, width, height, dark_count);
    CHECK(cudaDeviceSynchronize());
}

__global__ void kernel_mean_value(const float* __restrict__ image, uint64_t npixels, double *partial_sums) {
    extern __shared__ double shared_sum[];
    
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Load and sum
    shared_sum[tid] = 0.0;
    if (idx < npixels)
        shared_sum[tid] = (double)image[idx];
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            shared_sum[tid] += shared_sum[tid + s];
        __syncthreads();
    }
    
    // Write block result
    if (tid == 0)
        partial_sums[blockIdx.x] = shared_sum[0];
}

inline double cuda_mean_value(const float* __restrict__ d_image, uint64_t npixels) {
    int blocks = (npixels + 256 - 1) / 256;
    
    double *d_partial_sums;
    CHECK(cudaMalloc(&d_partial_sums, sizeof(double) * blocks));
    
    kernel_mean_value<<<blocks, 256, sizeof(double) * 256>>>(d_image, npixels, d_partial_sums);
    CHECK(cudaDeviceSynchronize());
    
    // Copy partial sums to host and finish reduction
    double *h_partial_sums = new double[blocks];
    CHECK(cudaMemcpy(h_partial_sums, d_partial_sums, sizeof(double) * blocks, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_partial_sums));
    
    double total_sum = 0.0;
    for (int i = 0; i < blocks; i++)
        total_sum += h_partial_sums[i];
    delete[] h_partial_sums;
    
    return total_sum / npixels;
}

void masterFlat_gpu(const uint16_t* __restrict__ flat_all,
                    const float* __restrict__ master_bias,
                    float* __restrict__ master_flat,
                    uint64_t width, uint64_t height, int flat_count) {
    uint64_t npixels = width * height;

    dim3 block_size(512);
    dim3 grid_size((npixels + block_size.x - 1)/block_size.x);
    kernel_mean<<<grid_size, block_size>>>(flat_all, master_flat, width, height, flat_count);
    CHECK(cudaDeviceSynchronize());
}

__global__ void calibrateLights_kernel(const uint16_t* __restrict__ light_images,
                                       const float* __restrict__ master_bias,
                                       const float* __restrict__ master_dark,
                                       const float* __restrict__ master_flat,
                                       float master_flat_mean_val,
                                       uint16_t* __restrict__ calib_all,
                                       uint64_t npixels,
                                       int light_count) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= npixels)
        return;
    
    float to_subtract = master_dark[idx];
    float denominator = (master_flat[idx] - master_bias[idx]) / master_flat_mean_val;
    
    for (uint i = 0; i < light_count; i++) {
        float numerator = (float)light_images[i * npixels + idx] - to_subtract;
        calib_all[i * npixels + idx] = clamp_u16_from_f32(numerator / denominator);
    }
}

void calibrateLights_gpu(
    const uint16_t* __restrict__ light_all,
    const float* __restrict__ master_bias,
    const float* __restrict__ master_dark,
    const float* __restrict__ master_flat,
    uint16_t* __restrict__ calib_all,
    uint64_t width, uint64_t height, int light_count) 
{
    // Implementazione simile a masterDark, ma con sottrazione del master bias e del master dark,
    // e divisione per il master flat
    // Per ogni pixel di ogni immagine light: calibrazione = (light - master_bias - master_dark) / master_flat

    uint64_t npixels = width * height;
    dim3 block_size(512);
    dim3 grid_size((npixels + block_size.x - 1)/block_size.x);

    float flat_mean_val = cuda_mean_value(master_flat, npixels);

    calibrateLights_kernel<<<grid_size, block_size>>>(light_all, master_bias, master_dark, master_flat, flat_mean_val, calib_all, npixels, light_count);
    CHECK(cudaDeviceSynchronize());
}