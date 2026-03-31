#include <cuda_runtime.h>
#include <stdint.h>
#include "common/cuda_check.h"

// calcolo media di tutte le immagini escludendo i pixel con valore 0,
// output del valore calcolato nell'array finale
__host__ __device__ inline float clamp(float x, float min_val, float max_val) {
    return fminf(fmaxf(x, min_val), max_val);
}

__device__ inline void computeMean2_float(float **image, float *mean, u_int64_t idx1, u_int64_t idx2, int numImages) {
    int count1 = 0, count2 = 0;
    float acc1 = 0.0f, acc2 = 0.0f;
    for (int i = 0; i < numImages; i++) {
        float val1 = image[i][idx1];
        float val2 = image[i][idx2];
        if (val1 > 0.0f) {
            count1++;
            acc1 += val1;
        }
        if (val2 > 0.0f) {
            count2++;
            acc2 += val2;
        }
    }
    mean[idx1] = (count1 > 0) ? clamp(acc1 / (float)count1, 0.0f, 65535.0f) : 0.0f;
    mean[idx2] = (count2 > 0) ? clamp(acc2 / (float)count2, 0.0f, 65535.0f) : 0.0f;
}

__global__ void masterBias_kernel(float **bias_images, float *master_bias, long width, long height, int bias_count) {
    u_int64_t idx_global = blockIdx.x * blockDim.x + threadIdx.x;
    u_int64_t npixels = width * height;

    if (idx_global >= npixels) return;

    computeMean2_float(bias_images, master_bias, idx_global, idx_global, bias_count);
}

__global__ void meanSubtract_kernel(float **images, float *bias, float *result, long width, long height, int count) {
    u_int64_t idx_global = blockIdx.x * blockDim.x + threadIdx.x;
    u_int64_t npixels = width * height;

    if (idx_global >= npixels) return;

    // Per ogni pixel, sottrai il master bias e poi calcola la media escludendo i valori <= 0
    int c = 0;
    float acc = 0.0f;
    for (int i = 0; i < count; i++) {
        float val = images[i][idx_global] - bias[idx_global];
        if (val > 0.0f) {
            c++;
            acc += val;
        }
    }
    result[idx_global] = (c > 0) ? clamp(acc / (float)c, 0.0f, 65535.0f) : 0.0f;
}

__global__ void calibrateLights_kernel(float **light_images, float *master_bias, float *master_dark, float *master_flat, float *calib_all, long npixels, int light_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= light_count * npixels) return;
    int img = i / npixels;
    int p = i % npixels;
    float val = light_images[img][p] - master_bias[p] - master_dark[p];
    val = fmaxf(val, 0.0f);
    float flat = master_flat[p];
    calib_all[i] = (flat > 0.0f) ? clamp(val / flat, 0.0f, 65535.0f) : 0.0f;
}

void masterBias(float *bias_all, float *master_bias, long width, long height, int bias_count) {
    u_int64_t npixels = (u_int64_t)width * (u_int64_t)height;
    
    // Allocate and prepare bias image pointers on host
    float **bias_images_host = (float **)malloc(bias_count * sizeof(float *));
    for (int i = 0; i < bias_count; i++) {
        bias_images_host[i] = bias_all + i * npixels;
    }
    
    // Copy pointers to device
    float **bias_images_device;
    CHECK(cudaMalloc(&bias_images_device, bias_count * sizeof(float *)));
    CHECK(cudaMemcpy(bias_images_device, bias_images_host, bias_count * sizeof(float *), cudaMemcpyHostToDevice));
    
    dim3 block_size(512);
    dim3 grid_size((npixels + block_size.x - 1)/block_size.x);
    masterBias_kernel<<<grid_size, block_size>>>(bias_images_device, master_bias, width, height, bias_count);
    CHECK(cudaDeviceSynchronize());
    
    CHECK(cudaFree(bias_images_device));
    free(bias_images_host);
}

void masterDark(float *dark_all, float *master_bias, float *master_dark, long width, long height, int dark_count) {
    // Implementazione simile a masterBias, ma con sottrazione del master bias
    // e calcolo della media per ogni pixel

    // sottrarre a ogni pixel di ogni immagine dark il corrispondente pixel del master bias con kernel
    u_int64_t npixels = (u_int64_t)width * (u_int64_t)height;
    // Allocate and prepare dark image pointers on host
    float **dark_images_host = (float **)malloc(dark_count * sizeof(float *));
    for (int i = 0; i < dark_count; i++) {
        dark_images_host[i] = dark_all + i * npixels;
    }

    // Copy pointers to device
    float **dark_images_device;
    CHECK(cudaMalloc(&dark_images_device, dark_count * sizeof(float *)));
    CHECK(cudaMemcpy(dark_images_device, dark_images_host, dark_count * sizeof(float *), cudaMemcpyHostToDevice));

    // Kernel per sottrazione del master bias e calcolo della media
    dim3 block_size(512);
    dim3 grid_size((npixels + block_size.x - 1)/block_size.x);
    meanSubtract_kernel<<<grid_size, block_size>>>(dark_images_device, master_bias, master_dark, width, height, dark_count);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaFree(dark_images_device));
    free(dark_images_host);
}

void masterFlat(float *flat_all, float *master_bias, float *master_flat, long width, long height, int flat_count) {
    // Sottrazione del master bias e divisione per il master flat

    // sottrarre a ogni pixel di ogni immagine flat il corrispondente pixel del master bias con kernel
    u_int64_t npixels = (u_int64_t)width * (u_int64_t)height;
    // Allocate and prepare flat image pointers on host
    float **flat_images_host = (float **)malloc(flat_count * sizeof(float *));
    for (int i = 0; i < flat_count; i++) {
        flat_images_host[i] = flat_all + i * npixels;
    }

    // Copy pointers to device
    float **flat_images_device;
    CHECK(cudaMalloc(&flat_images_device, flat_count * sizeof(float *)));
    CHECK(cudaMemcpy(flat_images_device, flat_images_host, flat_count * sizeof(float *), cudaMemcpyHostToDevice));

    // Kernel per sottrazione del master bias e calcolo della media
    dim3 block_size(512);
    dim3 grid_size((npixels + block_size.x - 1)/block_size.x);
    meanSubtract_kernel<<<grid_size, block_size>>>(flat_images_device, master_bias, master_flat, width, height, flat_count);
    CHECK(cudaDeviceSynchronize());

    // Normalizzazione del master flat (dividere ogni pixel per il valore medio del master flat)
    // Calcolo del valore medio del master flat
    float sum = 0.0f;
    for (u_int64_t i = 0; i < npixels; i++) {
        sum += master_flat[i];
    }
    float mean_val = (sum > 0.0f) ? sum / (float)npixels : 0.0f;

    // Normalizzazione del master flat
    if (mean_val > 0.0f) {
        for (u_int64_t i = 0; i < npixels; i++) {
            float normalized = master_flat[i] / mean_val;
            master_flat[i] = clamp(normalized, 0.0f, 65535.0f);
        }
    } else {
        // Se mean_val == 0, lascia master_flat come è (tutti 0), per evitare divisione per zero
        // In calibrateLights, flat == 0 verrà gestito impostando calib a 0
    }

    CHECK(cudaFree(flat_images_device));
    free(flat_images_host);
}

__global__ void calibrateLights_kernel(float **light_images,
                                       float *master_bias,
                                       float *master_dark,
                                       float *master_flat,
                                       float *calib_all,
                                       u_int64_t npixels,
                                       int light_count) {
    u_int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    u_int64_t total = (u_int64_t)light_count * npixels;
    if (idx >= total) return;

    u_int64_t pixel = idx % npixels;
    u_int64_t img_idx = idx / npixels;

    float val = light_images[img_idx][pixel] - master_bias[pixel] - master_dark[pixel];
    val = fmaxf(val, 0.0f);

    float flat = master_flat[pixel];
    if (flat > 0.0f) {
        float result = val / flat;
        calib_all[idx] = clamp(result, 0.0f, 65535.0f);
    } else {
        calib_all[idx] = 0.0f;
    }
}

void calibrateLights(float *light_all, float *master_bias, float *master_dark, float *master_flat, float *calib_all, long width, long height, int light_count) {
    // Implementazione simile a masterDark, ma con sottrazione del master bias e del master dark,
    // e divisione per il master flat
    // Per ogni pixel di ogni immagine light: calibrazione = (light - master_bias - master_dark) / master_flat

    u_int64_t npixels = (u_int64_t)width * (u_int64_t)height;
    // Allocate and prepare light image pointers on host
    float **light_images_host = (float **)malloc(light_count * sizeof(float *));
    for (int i = 0; i < light_count; i++) {
        light_images_host[i] = light_all + i * npixels;
    }

    // Copy pointers to device
    float **light_images_device;
    CHECK(cudaMalloc(&light_images_device, light_count * sizeof(float *)));
    CHECK(cudaMemcpy(light_images_device, light_images_host, light_count * sizeof(float *), cudaMemcpyHostToDevice));

    dim3 block_size(512);
    dim3 grid_size(((u_int64_t)light_count * npixels + block_size.x - 1)/block_size.x);
    calibrateLights_kernel<<<grid_size, block_size>>>(light_images_device, master_bias, master_dark, master_flat, calib_all, npixels, light_count);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaFree(light_images_device));
    free(light_images_host);
}