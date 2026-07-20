#ifndef HOST_ALFA_SIGMA_H
#define HOST_ALFA_SIGMA_H

#include <stdio.h>
#include <fitsio.h>
#include <dirent.h>
#include <string.h>
#include <math.h>

void mean_cpu(u_int16_t* __restrict__ img_all, u_int16_t* __restrict__ mean, int numImages, int npixels) {
    #pragma omp parallel for
    for (int i = 0; i < npixels; i++) {
        u_int16_t count = 0;
        u_int32_t acc = 0;
        for (int j = 0; j < numImages; j++) {
            u_int64_t base = (u_int64_t)j * npixels;
            u_int16_t val = img_all[base + i];
            if (val > 0) {
                count++;
                acc += val;
            }
        }
        mean[i] = (count > 0) ? acc / count : 0;
    }
}

void std_dev_cpu(float* __restrict__ std, u_int16_t* __restrict__ mean, u_int16_t* __restrict__ img_all, int numImages, int npixels) {
    
    #pragma omp parallel for
    for (int i = 0; i < npixels; i++) {
        u_int16_t count = 0;
        std[i] = 0.0f;
        for (int j = 0; j < numImages; j++) {
            u_int64_t base = (u_int64_t)j * npixels;
            u_int16_t val = img_all[base + i];
            if (val > 0) {
                count++;
                std[i] += ((float)val - mean[i]) * ((float)val - mean[i]);
            }
        }
        if (count > 1)
            std[i] = sqrtf(std[i] / count);
    }
}

void filter_pixels_cpu(u_int16_t* __restrict__ mean, float* __restrict__ std, u_int16_t* __restrict__ img_all, float k, int numImages, int npixels) {
    float k2 = k * k;

    #pragma omp parallel for
    for (int i = 0; i < npixels; i++) {
        float s2 = k2 * std[i] * std[i];  // k^2 * variance (std contains stddev, so square it)
        for (int j = 0; j < numImages; j++) {
            u_int64_t base = (u_int64_t)j * npixels;
            u_int16_t val = img_all[base + i];
            if (val > 0) {
                float d = (float)val - (float)mean[i];
                if (d * d > s2)
                    img_all[base + i] = 0;
            }
        }
    }
}

void alfa_sigma_cpu(
    u_int16_t* __restrict__ img_all,
    u_int16_t* __restrict__ mean,
    float* __restrict__ std,
    int numImages,
    int npixels,
    float kappa,
    int sigma
) {
    for (int i = 0; i < sigma; i++) {
        mean_cpu(img_all, mean, numImages, npixels);
        std_dev_cpu(std, mean, img_all, numImages, npixels);
        filter_pixels_cpu(mean, std, img_all, kappa, numImages, npixels);
    }
    mean_cpu(img_all, mean, numImages, npixels);
}

#endif // HOST_ALFA_SIGMA_H