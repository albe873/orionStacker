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

__device__ inline void computeMean2_float(u_int16_t **image, u_int16_t *mean, u_int64_t idx1, u_int64_t idx2, int numImages) {
    int count1 = 0, count2 = 0;
    float acc1 = 0.0f, acc2 = 0.0f;
    for (int i = 0; i < numImages; i++) {
        auto val1 = image[i][idx1];
        auto val2 = image[i][idx2];
        if (val1 > 0) {
            count1++;
            acc1 += val1;
        }
        if (val2 > 0) {
            count2++;
            acc2 += val2;
        }
    }
    mean[idx1] = (count1 > 0) ? clamp_u16_from_f32(acc1 / (float)count1) : 0;
    mean[idx2] = (count2 > 0) ? clamp_u16_from_f32(acc2 / (float)count2) : 0;
}

__global__ void masterBias_kernel(u_int16_t **bias_images, u_int16_t *master_bias, long width, long height, int bias_count) {
    u_int64_t idx_global = blockIdx.x * blockDim.x + threadIdx.x;
    u_int64_t npixels = width * height;

    if (idx_global >= npixels) return;

    computeMean2_float(bias_images, master_bias, idx_global, idx_global, bias_count);
}

__global__ void meanSubtract_kernel(u_int16_t **images, u_int16_t *bias, u_int16_t *result, long width, long height, int count) {
    u_int64_t idx_global = blockIdx.x * blockDim.x + threadIdx.x;
    u_int64_t npixels = width * height;

    if (idx_global >= npixels) return;

    // Per ogni pixel, sottrai il master bias e poi calcola la media escludendo i valori <= 0
    int c = 0;
    float acc = 0.0f;
    for (int i = 0; i < count; i++) {
        float val = (float)images[i][idx_global] - (float)bias[idx_global];
        if (val > 0) {
            c++;
            acc += val;
        }
    }
    result[idx_global] = (c > 0) ? clamp_u16_from_f32(acc / (float)c) : 0;
}

void masterBias(u_int16_t *bias_all, u_int16_t *master_bias, long width, long height, int bias_count) {
    u_int64_t npixels = (u_int64_t)width * (u_int64_t)height;
    
    // Allocate and prepare bias image pointers on host
    u_int16_t **bias_images_host = (u_int16_t **)malloc(bias_count * sizeof(u_int16_t *));
    for (int i = 0; i < bias_count; i++) {
        bias_images_host[i] = bias_all + i * npixels;
    }
    
    // Copy pointers to device
    u_int16_t **bias_images_device;
    CHECK(cudaMalloc(&bias_images_device, bias_count * sizeof(u_int16_t *)));
    CHECK(cudaMemcpy(bias_images_device, bias_images_host, bias_count * sizeof(u_int16_t *), cudaMemcpyHostToDevice));
    
    dim3 block_size(512);
    dim3 grid_size((npixels + block_size.x - 1)/block_size.x);
    masterBias_kernel<<<grid_size, block_size>>>(bias_images_device, master_bias, width, height, bias_count);
    CHECK(cudaDeviceSynchronize());
    
    CHECK(cudaFree(bias_images_device));
    free(bias_images_host);
}

void masterDark(u_int16_t *dark_all, u_int16_t *master_bias, u_int16_t *master_dark, long width, long height, int dark_count) {
    // Implementazione simile a masterBias, ma con sottrazione del master bias
    // e calcolo della media per ogni pixel

    // sottrarre a ogni pixel di ogni immagine dark il corrispondente pixel del master bias con kernel
    u_int64_t npixels = (u_int64_t)width * (u_int64_t)height;
    // Allocate and prepare dark image pointers on host
    u_int16_t **dark_images_host = (u_int16_t **)malloc(dark_count * sizeof(u_int16_t *));
    for (int i = 0; i < dark_count; i++) {
        dark_images_host[i] = dark_all + i * npixels;
    }

    // Copy pointers to device
    u_int16_t **dark_images_device;
    CHECK(cudaMalloc(&dark_images_device, dark_count * sizeof(u_int16_t *)));
    CHECK(cudaMemcpy(dark_images_device, dark_images_host, dark_count * sizeof(u_int16_t *), cudaMemcpyHostToDevice));

    // Kernel per sottrazione del master bias e calcolo della media
    dim3 block_size(512);
    dim3 grid_size((npixels + block_size.x - 1)/block_size.x);
    meanSubtract_kernel<<<grid_size, block_size>>>(dark_images_device, master_bias, master_dark, width, height, dark_count);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaFree(dark_images_device));
    free(dark_images_host);
}

void masterFlat(u_int16_t *flat_all, u_int16_t *master_bias, u_int16_t *master_flat, long width, long height, int flat_count) {
    // Sottrazione del master bias e divisione per il master flat

    // sottrarre a ogni pixel di ogni immagine flat il corrispondente pixel del master bias con kernel
    u_int64_t npixels = (u_int64_t)width * (u_int64_t)height;
    // Allocate and prepare flat image pointers on host
    u_int16_t **flat_images_host = (u_int16_t **)malloc(flat_count * sizeof(u_int16_t *));
    for (int i = 0; i < flat_count; i++) {
        flat_images_host[i] = flat_all + i * npixels;
    }

    // Copy pointers to device
    u_int16_t **flat_images_device;
    CHECK(cudaMalloc(&flat_images_device, flat_count * sizeof(u_int16_t *)));
    CHECK(cudaMemcpy(flat_images_device, flat_images_host, flat_count * sizeof(u_int16_t *), cudaMemcpyHostToDevice));

    // Kernel per sottrazione del master bias e calcolo della media
    dim3 block_size(512);
    dim3 grid_size((npixels + block_size.x - 1)/block_size.x);
    meanSubtract_kernel<<<grid_size, block_size>>>(flat_images_device, master_bias, master_flat, width, height, flat_count);
    CHECK(cudaDeviceSynchronize());

    // Normalizzazione del master flat (dividere ogni pixel per il valore medio del master flat)
    // Calcolo del valore medio del master flat
    u_int64_t sum = 0;
    for (u_int64_t i = 0; i < npixels; i++) {
        sum += master_flat[i];
    }
    float mean_val = (sum > 0) ? ((float)sum / (float)npixels) : 0.0f;

    // Normalizzazione del master flat
    if (mean_val > 0.0f) {
        const float norm_scale = 65535.0f / mean_val;
        for (u_int64_t i = 0; i < npixels; i++) {
            // Store normalized flat in fixed-point scale [0, 65535].
            float scaled = (float)master_flat[i] * norm_scale;
            master_flat[i] = clamp_u16_from_f32(scaled);
        }
    } else {
        // Se mean_val == 0, lascia master_flat come è (tutti 0), per evitare divisione per zero
        // In calibrateLights, flat == 0 verrà gestito impostando calib a 0
    }

    CHECK(cudaFree(flat_images_device));
    free(flat_images_host);
}

__global__ void calibrateLights_kernel(u_int16_t **light_images,
                                       u_int16_t *master_bias,
                                       u_int16_t *master_dark,
                                       u_int16_t *master_flat,
                                       u_int16_t *calib_all,
                                       u_int64_t npixels,
                                       int light_count) {
    u_int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    u_int64_t total = (u_int64_t)light_count * npixels;
    if (idx >= total) return;

    u_int64_t pixel = idx % npixels;
    u_int64_t img_idx = idx / npixels;

    float val = (float)light_images[img_idx][pixel] - (float)master_bias[pixel] - (float)master_dark[pixel];
    if (val < 0.0f) val = 0.0f;

    u_int16_t flat = master_flat[pixel];
    if (flat > 0) {
        // master_flat is normalized in [0, 65535], so multiply by 65535 before division.
        float scaled = (val * 65535.0f) / (float)flat;
        calib_all[idx] = clamp_u16_from_f32(scaled);
    } else {
        calib_all[idx] = 0;
    }
}

void calibrateLights(u_int16_t *light_all, u_int16_t *master_bias, u_int16_t *master_dark, u_int16_t *master_flat, u_int16_t *calib_all, long width, long height, int light_count) {
    // Implementazione simile a masterDark, ma con sottrazione del master bias e del master dark,
    // e divisione per il master flat
    // Per ogni pixel di ogni immagine light: calibrazione = (light - master_bias - master_dark) / master_flat

    u_int64_t npixels = (u_int64_t)width * (u_int64_t)height;
    // Allocate and prepare light image pointers on host
    u_int16_t **light_images_host = (u_int16_t **)malloc(light_count * sizeof(u_int16_t *));
    for (int i = 0; i < light_count; i++) {
        light_images_host[i] = light_all + i * npixels;
    }

    // Copy pointers to device
    u_int16_t **light_images_device;
    CHECK(cudaMalloc(&light_images_device, light_count * sizeof(u_int16_t *)));
    CHECK(cudaMemcpy(light_images_device, light_images_host, light_count * sizeof(u_int16_t *), cudaMemcpyHostToDevice));

    dim3 block_size(512);
    dim3 grid_size(((u_int64_t)light_count * npixels + block_size.x - 1)/block_size.x);
    calibrateLights_kernel<<<grid_size, block_size>>>(light_images_device, master_bias, master_dark, master_flat, calib_all, npixels, light_count);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaFree(light_images_device));
    free(light_images_host);
}