#include <cuda_runtime.h>
#include <stdint.h>
#include "common/cuda_check.h"

// calcolo media di tutte le immagini escludendo i pixel con valore 0,
// output del valore calcolato nell'array finale
__device__ inline void computeMean2_uint16(u_int16_t **image, u_int16_t *mean, u_int64_t idx1, u_int64_t idx2, u_int16_t numImages) {
    u_int16_t count1 = 0, count2 = 0;
    u_int32_t acc1 = 0, acc2 = 0;
    for (int i = 0; i < numImages; i++) {
        u_int16_t val1 = image[i][idx1];
        u_int16_t val2 = image[i][idx2];
        if (val1 > 0) {
            count1++;
            acc1 += val1;
        }
        if (val2 > 0) {
            count2++;
            acc2 += val2;
        }
    }
    mean[idx1] = (count1 > 0) ? acc1 / count1 : 0;
    mean[idx2] = (count2 > 0) ? acc2 / count2 : 0;
}

__global__ void masterBias_kernel(u_int16_t **bias_images, u_int16_t *master_bias, long width, long height, int bias_count) {
    u_int64_t idx_global = blockIdx.x * blockDim.x + threadIdx.x;
    u_int64_t npixels = width * height;

    if (idx_global >= npixels) return;

    computeMean2_uint16(bias_images, master_bias, idx_global, idx_global, bias_count);
}

__global__ void meanSubtract_kernel(u_int16_t **images, u_int16_t *bias, u_int16_t *result, long width, long height, int count) {
    u_int64_t idx_global = blockIdx.x * blockDim.x + threadIdx.x;
    u_int64_t npixels = width * height;

    if (idx_global >= npixels) return;

    // Per ogni pixel, sottrai il master bias e poi calcola la media escludendo i valori <= 0
    u_int16_t c = 0;
    u_int32_t acc = 0;
    for (int i = 0; i < count; i++) {
        int val = (int)images[i][idx_global] - (int)bias[idx_global];
        if (val > 0) {
            c++;
            acc += val;
        }
    }
    result[idx_global] = (c > 0) ? acc / c : 0;
}

__global__ void calibrateLights_kernel(u_int16_t **light_images, u_int16_t *master_bias, u_int16_t *master_dark, u_int16_t *master_flat, u_int16_t *calib_all, long npixels, int light_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= light_count * npixels) return;
    int img = i / npixels;
    int p = i % npixels;
    int val = (int)light_images[img][p] - (int)master_bias[p] - (int)master_dark[p];
    if (val < 0) val = 0;
    const int flat = master_flat[p];
    calib_all[i] = (flat > 0) ? (u_int16_t)(val / flat) : 0;
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
    u_int64_t mean = (sum > 0) ? sum / npixels : 0;

    // Normalizzazione del master flat
    for (u_int64_t i = 0; i < npixels; i++) {
        master_flat[i] = (mean > 0) ? master_flat[i] / mean : 0;
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

    int val = (int)light_images[img_idx][pixel] - (int)master_bias[pixel] - (int)master_dark[pixel];
    if (val < 0) val = 0;

    u_int16_t flat = master_flat[pixel];
    if (flat > 0) {
        calib_all[idx] = (u_int16_t)(val / flat);
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