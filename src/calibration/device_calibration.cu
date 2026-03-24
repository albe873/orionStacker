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