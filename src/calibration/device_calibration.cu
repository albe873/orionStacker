#include <cuda_runtime.h>
#include <stdint.h>
#include <algorithm>
#include "cuda_helper.h"

// calcolo media di tutte le immagini escludendo i pixel con valore 0,
// output del valore calcolato nell'array finale
__host__ __device__ inline u_int16_t clamp(u_int16_t x, u_int16_t min_val, u_int16_t max_val) {
    return std::min<u_int16_t>(std::max<u_int16_t>(x, min_val), max_val);
}

__host__ __device__ inline u_int16_t clamp_u16_from_u64(u_int64_t x) {
    return (x > 65535ULL) ? 65535 : (u_int16_t)x;
}

__host__ __device__ inline u_int16_t clamp_u16_from_f32(float x) {
    if (x <= 0.0f) return 0;
    if (x >= 65535.0f) return 65535;
    return (u_int16_t)(x + 0.5f);
}


__global__ void masterBias_kernel(const u_int16_t* __restrict__ bias_images,
                                  u_int16_t* __restrict__ master_bias,
                                  long width, long height, int bias_count) {
    u_int64_t idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    u_int64_t idx2 = idx1 + 1;
    u_int64_t npixels = width * height;
    if (idx2 >= npixels)
        return;

    int count1 = 0, count2 = 0;
    float acc1 = 0.0f, acc2 = 0.0f;
    for (int i = 0; i < bias_count; i++) {
        auto val1 = bias_images[i * npixels + idx1];
        auto val2 = bias_images[i * npixels + idx2];
        if (val1 > 0) {
            count1++;
            acc1 += val1;
        }
        if (val2 > 0) {
            count2++;
            acc2 += val2;
        }
    }
    master_bias[idx1] = (count1 > 0) ? clamp_u16_from_f32(acc1 / (float)count1) : 0;
    master_bias[idx2] = (count2 > 0) ? clamp_u16_from_f32(acc2 / (float)count2) : 0;
    
}

void masterBias(const u_int16_t* __restrict__ bias_all,
                u_int16_t* __restrict__ master_bias,
                long width, long height, int bias_count) {
    u_int64_t npixels = (u_int64_t)width * (u_int64_t)height;
    
    dim3 block_size(512);
    dim3 grid_size((npixels + block_size.x - 1)/block_size.x);
    masterBias_kernel<<<grid_size, block_size>>>(bias_all, master_bias, width, height, bias_count);
    CHECK(cudaDeviceSynchronize());
}




__global__ void meanSubtract_kernel(const u_int16_t* __restrict__ images,
                                    const u_int16_t* __restrict__ bias,
                                    u_int16_t* __restrict__ result,
                                    long width, long height, int count) {
    u_int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    u_int64_t npixels = width * height;

    if (idx >= npixels)
        return;

    // Per ogni pixel, sottrai il master bias e poi calcola la media escludendo i valori <= 0
    int c = 0;
    float acc = 0.0f;
    for (int i = 0; i < count; i++) {
        float val = (float)images[i * npixels + idx] - (float)bias[idx];
        if (val > 0) {
            c++;
            acc += val;
        }
    }
    result[idx] = (c > 0) ? clamp_u16_from_f32(acc / (float)c) : 0;
}

void masterDark(const u_int16_t* __restrict__ dark_all,
                const u_int16_t* __restrict__ master_bias,
                u_int16_t* __restrict__ master_dark,
                long width, long height, int dark_count) {
    // Implementazione simile a masterBias, ma con sottrazione del master bias
    // e calcolo della media per ogni pixel

    // sottrarre a ogni pixel di ogni immagine dark il corrispondente pixel del master bias con kernel
    u_int64_t npixels = (u_int64_t)width * (u_int64_t)height;

    // Kernel per sottrazione del master bias e calcolo della media
    dim3 block_size(512);
    dim3 grid_size((npixels + block_size.x - 1)/block_size.x);
    meanSubtract_kernel<<<grid_size, block_size>>>(dark_all, master_bias, master_dark, width, height, dark_count);
    CHECK(cudaDeviceSynchronize());
}

__global__ void kernel_calculate_mean(const u_int16_t* __restrict__ image, u_int64_t npixels, double *partial_sums) {
    extern __shared__ double shared_sum[];
    
    u_int64_t idx = (u_int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Load and sum
    shared_sum[tid] = 0.0;
    if (idx < npixels) {
        shared_sum[tid] = (double)image[idx];
    }
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    // Write block result
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_sum[0];
    }
}

__global__ void kernel_normalize(u_int16_t* __restrict__ image, u_int64_t npixels, float norm_scale) {
    u_int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= npixels)
        return;

    float scaled = (float)image[idx] * norm_scale;
    image[idx] = clamp_u16_from_f32(scaled);
}

inline double cuda_calculate_mean(const u_int16_t* __restrict__ d_image, u_int64_t npixels) {
    int blocks = (npixels + 256 - 1) / 256;
    
    double *d_partial_sums;
    CHECK(cudaMalloc(&d_partial_sums, sizeof(double) * blocks));
    
    kernel_calculate_mean<<<blocks, 256, sizeof(double) * 256>>>(d_image, npixels, d_partial_sums);
    CHECK(cudaDeviceSynchronize());
    
    // Copy partial sums to host and finish reduction
    double *h_partial_sums = new double[blocks];
    CHECK(cudaMemcpy(h_partial_sums, d_partial_sums, sizeof(double) * blocks, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_partial_sums));
    
    double total_sum = 0.0;
    for (int i = 0; i < blocks; i++) {
        total_sum += h_partial_sums[i];
    }
    delete[] h_partial_sums;
    
    return total_sum / (double)npixels;
}

void masterFlat(const u_int16_t* __restrict__ flat_all,
                const u_int16_t* __restrict__ master_bias,
                u_int16_t* __restrict__ master_flat,
                long width, long height, int flat_count) {
    // Sottrazione del master bias e divisione per il master flat

    // sottrarre a ogni pixel di ogni immagine flat il corrispondente pixel del master bias con kernel
    u_int64_t npixels = (u_int64_t)width * (u_int64_t)height;

    // Kernel per sottrazione del master bias e calcolo della media
    dim3 block_size(512);
    dim3 grid_size((npixels + block_size.x - 1)/block_size.x);
    meanSubtract_kernel<<<grid_size, block_size>>>(flat_all, master_bias, master_flat, width, height, flat_count);
    CHECK(cudaDeviceSynchronize());

    // Normalizzazione del master flat (dividere ogni pixel per il valore medio del master flat)
    // Calcolo del valore medio del master flat

    float mean_val = cuda_calculate_mean(master_flat, npixels);

    // Normalizzazione del master flat
    // Se mean_val == 0, lascia master_flat come è (tutti 0), per evitare divisione per zero
    // In calibrateLights, flat == 0 verrà gestito impostando calib a 0
    if (mean_val > 0.0f) {
        const float norm_scale = 65535.0f / mean_val;
        kernel_normalize<<<grid_size, block_size>>>(master_flat, npixels, norm_scale);
        CHECK(cudaDeviceSynchronize());
    }
}

__global__ void calibrateLights_kernel(const u_int16_t* __restrict__ light_images,
                                       const u_int16_t* __restrict__ master_bias,
                                       const u_int16_t* __restrict__ master_dark,
                                       const u_int16_t* __restrict__ master_flat,
                                       u_int16_t* __restrict__ calib_all,
                                       u_int64_t npixels,
                                       int light_count) {
    u_int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= npixels)
        return;
    
    float to_subtract = (float)master_bias[idx] + (float)master_dark[idx];
    u_int16_t flat_val = master_flat[idx];
    
    for (uint i = 0; i < light_count; i++) {
        float val = (float)light_images[i * npixels + idx] - to_subtract;
        if (val < 0.0f)
            val = 0.0f;

        if (flat_val > 0) {
            float scaled = (val * 65535.0f) / (float)flat_val;
            calib_all[i * npixels + idx] = clamp_u16_from_f32(scaled);
        } else {
            calib_all[i * npixels + idx] = 0;
        }
    }
}

void calibrateLights(
    const u_int16_t* __restrict__ light_all,
    const u_int16_t* __restrict__ master_bias,
    const u_int16_t* __restrict__ master_dark,
    const u_int16_t* __restrict__ master_flat,
    u_int16_t* __restrict__ calib_all,
    long width, long height, int light_count) 
{
    // Implementazione simile a masterDark, ma con sottrazione del master bias e del master dark,
    // e divisione per il master flat
    // Per ogni pixel di ogni immagine light: calibrazione = (light - master_bias - master_dark) / master_flat

    u_int64_t npixels = (u_int64_t)width * (u_int64_t)height;

    dim3 block_size(512);
    dim3 grid_size((npixels + block_size.x - 1)/block_size.x);
    calibrateLights_kernel<<<grid_size, block_size>>>(light_all, master_bias, master_dark, master_flat, calib_all, npixels, light_count);
    CHECK(cudaDeviceSynchronize());
}