#include "cuda_helper.h"

// -------------- uint16_t version --------------

// funzioni con versioni per singolo pixel e per 2 pixel
// calcolo 2 pixel per thread per migliorare l'efficacia della cache
// (linee da 128 byte: 2 (pixel) * 2 (byte per pixel) * 32 (warp size) = 128 byte)

// calcolo media di tutte le immagini escludendo i pixel con valore 0,
// output del valore calcolato nell'array finale

__device__ inline void mean_2_uint16(u_int16_t *img_all, u_int16_t *mean, 
                                     u_int64_t idx1, u_int64_t idx2, 
                                     u_int16_t numImages, u_int64_t npixels) {
    u_int16_t count1 = 0, count2 = 0;
    u_int32_t acc1 = 0, acc2 = 0;
    for (int i = 0; i < numImages; i++) {
        u_int64_t base = (u_int64_t)i * npixels;
        auto val1 = img_all[base + idx1];
        auto val2 = img_all[base + idx2];
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

__device__ inline void mean_uint16(u_int16_t *img_all, u_int16_t *mean, 
                                   u_int64_t idx, 
                                   u_int16_t numImages, u_int64_t npixels) {
    u_int16_t count = 0;
    u_int32_t acc = 0;
    for (int i = 0; i < numImages; i++) {
        u_int64_t base = (u_int64_t)i * npixels;
        u_int16_t val = img_all[base + idx];
        if (val > 0) {
            count++;
            acc += val;
        }
    }
    mean[idx] = (count > 0) ? acc / count : 0;
}

__device__ inline void partial_mean_2_uint16(u_int16_t *img_all, 
                                             u_int16_t* mean1, u_int16_t* mean2,
                                             u_int64_t idx1, u_int64_t idx2,
                                             u_int16_t numImages, u_int64_t npixels) {
    u_int32_t acc1 = 0, acc2 = 0;
    u_int16_t count1 = 0, count2 = 0;
    for (int i = 0; i < numImages; i++) {
        u_int64_t base = (u_int64_t)i * npixels;
        auto val1 = img_all[base + idx1];
        auto val2 = img_all[base + idx2];
        if (val1 > 0) {
            acc1 += val1;
            count1++;
        }
        if (val2 > 0) {
            acc2 += val2;
            count2++;
        }
    }
    *mean1 = (count1 > 0) ? acc1 / count1 : 0;
    *mean2 = (count2 > 0) ? acc2 / count2 : 0;
}

__device__ inline void partial_mean_uint16(u_int16_t *img_all, u_int16_t* mean,
                                           u_int64_t idx,
                                           u_int16_t numImages, u_int64_t npixels) {
    u_int32_t acc = 0;
    u_int16_t count = 0;
    for (int i = 0; i < numImages; i++) {
        u_int64_t base = (u_int64_t)i * npixels;
        auto val = img_all[base + idx];
        if (val > 0) {
            acc += val;
            count++;
        }
    }
    *mean = (count > 0) ? acc / count : 0;
}

__device__ inline void std_dev_2_uint16(float *std1, float *std2, 
                                        u_int16_t mean1, u_int16_t mean2,
                                        u_int16_t *img_all,
                                        u_int64_t idx1, u_int64_t idx2,
                                        u_int16_t numImages, u_int64_t npixels) {
    u_int16_t count1 = 0, count2 = 0;
    *std1 = 0.0f;
    *std2 = 0.0f;
    for (int i = 0; i < numImages; i++) {
        u_int64_t base = (u_int64_t)i * npixels;
        auto val1 = img_all[base + idx1];
        auto val2 = img_all[base + idx2];
        if (val1 > 0) {
            count1++;
            *std1 += ((float)val1 - mean1) * ((float)val1 - mean1);
        }
        if (val2 > 0) {
            count2++;
            *std2 += ((float)val2 - mean2) * ((float)val2 - mean2);
        }
    }
    *std1 = (count1 > 1) ? (*std1 / count1) : 0.0f;
    *std2 = (count2 > 1) ? (*std2 / count2) : 0.0f;
}

__device__ inline void std_dev_uint16(float *std, u_int16_t mean,
                                      u_int16_t *img_all,
                                      u_int64_t idx,
                                      u_int16_t numImages, u_int64_t npixels) {
    u_int16_t count = 0;
    *std = 0.0f;
    for (int i = 0; i < numImages; i++) {
        u_int64_t base = (u_int64_t)i * npixels;
        auto val = img_all[base + idx];
        if (val > 0) {
            count++;
            *std += ((float)val - mean) * ((float)val - mean);
        }
    }
    *std = (count > 1) ? (*std / count) : 0.0f;
}

__device__ inline void filter_pixels_2_uint16(u_int16_t mean1, float std1, 
                                              u_int16_t mean2, float std2,
                                              u_int16_t *img_all,
                                              u_int64_t idx1, u_int64_t idx2,
                                              float k, 
                                              u_int16_t numImages, u_int64_t npixels) {
    const float k2 = k * k;
    for (int i = 0; i < numImages; i++) {
        u_int64_t base = (u_int64_t)i * npixels;
        auto val1 = img_all[base + idx1];
        auto val2 = img_all[base + idx2];

        if (val1 > 0) {
            float d1 = (float)val1 - (float)mean1;
            if (d1 * d1 > k2 * std1)
                img_all[base + idx1] = 0;
        }

        if (val2 > 0) {
            float d2 = (float)val2 - (float)mean2;
            if (d2 * d2 > k2 * std2)
                img_all[base + idx2] = 0;
        }
    }
}

__device__ inline void filter_pixels_uint16(u_int16_t mean, float std,
                                            u_int16_t *img_all,
                                            u_int64_t idx,
                                            float k,
                                            u_int16_t numImages, u_int64_t npixels) {
    const float k2 = k * k;
    for (int i = 0; i < numImages; i++) {
        u_int64_t base = (u_int64_t)i * npixels;
        u_int16_t val = img_all[base + idx];
        if (val > 0) {
            float d = (float)val - (float)mean;
            if (d * d > k2 * std)
                img_all[base + idx] = 0;
        }
    }
}

__global__ void kernel_alfa_sigma_uint16(u_int16_t *img_all, u_int16_t *mean, 
                                         const u_int16_t numImages, const u_int64_t npixels, 
                                         const float k, const u_int16_t s) {
    const u_int64_t idx1 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const u_int64_t idx2 = idx1 + 1;

    float std1, std2;
    u_int16_t part_mean1, part_mean2;
    
    if (idx2 < npixels) {
        for (u_int16_t i = 0; i < s; i++) {
            partial_mean_2_uint16(img_all, &part_mean1, &part_mean2, idx1, idx2, numImages, npixels);
            std_dev_2_uint16(&std1, &std2, part_mean1, part_mean2, img_all, idx1, idx2, numImages, npixels);
            filter_pixels_2_uint16(part_mean1, std1, part_mean2, std2, img_all, idx1, idx2, k, numImages, npixels);
        }
        mean_2_uint16(img_all, mean, idx1, idx2, numImages, npixels);
    }

    if (idx2 == npixels) { // caso dispari
        for (u_int16_t i = 0; i < s; i++) {
            partial_mean_uint16(img_all, &part_mean1, idx1, numImages, npixels);
            std_dev_uint16(&std1, part_mean1, img_all, idx1, numImages, npixels);
            filter_pixels_uint16(part_mean1, std1, img_all, idx1, k, numImages, npixels);
        }
        mean_uint16(img_all, mean, idx1, numImages, npixels);
    }
}

void alfa_sigma(u_int16_t *img_all, u_int16_t *mean, 
                const u_int16_t numImages, const u_int64_t npixels, 
                const float k, const u_int16_t s) {
    dim3 block_size(512);
    dim3 grid_size = (npixels / 2 + block_size.x - 1) / block_size.x;

    kernel_alfa_sigma_uint16<<<grid_size, block_size>>>(img_all, mean, numImages, npixels, k, s);
    CHECK(cudaDeviceSynchronize());
}