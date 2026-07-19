#ifndef HOST_ALFA_SIGMA_H
#define HOST_ALFA_SIGMA_H

#include <stdio.h>
#include <fitsio.h>
#include <dirent.h>
#include <string.h>
#include <math.h>

void accumulatePixelsCPU(u_int32_t* __restrict__ acc, u_int16_t* __restrict__ image, int npixels) {
    for (int i = 0; i < npixels; i++) {
        acc[i] += image[i];
    }
}
void computeMeanCPU(u_int16_t** __restrict__ image, u_int16_t* __restrict__ mean, int numImages, int npixels) {
    for (int i = 0; i < npixels; i++) {
        u_int16_t immagini = 0;
        u_int32_t acc = 0;
        for (int j = 0; j < numImages; j++) {
            if (image[j][i] > 0) {
                immagini++;
                acc += image[j][i];
            }
        }
        if (immagini > 0)
            mean[i] = acc / immagini;
        else
            mean[i] = 0;
    }
}
void computeStdDevCPU(float* __restrict__ std, u_int16_t* __restrict__ mean, u_int16_t** __restrict__ image,
                      int numImages, int npixels) {
    u_int16_t immagini;
    for (int i = 0; i < npixels; i++) {
        immagini = 0;
        std[i] = 0;
        for (int j = 0; j < numImages; j++) {
            if (image[j][i] > 0) {
                immagini++;
                std[i] += ((float) image[j][i] - mean[i]) * ((float) image[j][i] - mean[i]);
            }
        }
        if (immagini > 1)
            std[i] = sqrtf(std[i] / immagini);
    }
}
void filterPixelsCPU(u_int16_t* __restrict__ mean, float* __restrict__ std, u_int16_t** __restrict__ image,
                     int k, int numImages, int npixels) {
    for (int i = 0; i < npixels; i++) {
        for (int j = 0; j < numImages; j++) {
            if (image[j][i] > mean[i] + k * std[i] || image[j][i] < mean[i] - k * std[i]) {
                image[j][i] = 0;
            }
        }
    }
}


void compute_alfa_sigma_cpu(
    u_int16_t** __restrict__ image,
    u_int16_t* __restrict__ mean,
    float* __restrict__ std,
    int numImages,
    int npixels,
    float kappa,
    int sigma
) {
    for (int i = 0; i < sigma; i++) {
        computeMeanCPU(image, mean, numImages, npixels);
        computeStdDevCPU(std, mean, image, numImages, npixels);
        filterPixelsCPU(mean, std, image, kappa, numImages, npixels);
    }
    computeMeanCPU(image, mean, numImages, npixels);
}

#endif // HOST_ALFA_SIGMA_H