#ifndef HOST_ALFA_SIGMA_H
#define HOST_ALFA_SIGMA_H

#include <stdio.h>
#include <fitsio.h>
#include <dirent.h>
#include <string.h>
#include <math.h>

// controllo risultato CPU-GPU media standard
void accumulatePixelsCPU(u_int32_t *acc, u_int16_t *image, int npixels) {
    for (int i = 0; i < npixels; i++) {
        acc[i] += image[i];
    }
}
void computeMeanCPU(u_int16_t **image, u_int16_t *mean, int numImages, int npixels) {
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
void computeStdDevCPU(float *std, u_int16_t *mean, u_int16_t **image, int numImages, int npixels) {
    u_int16_t immagini;
    for (int i = 0; i < npixels; i++) {
        immagini = 0;
        //std[i] = 0.0f;    // inizializzato a 0.0f, più veloce impostare tutto a zero con cudaMemset
        for (int j = 0; j < numImages; j++) {
            if (image[j][i] > 0) {
                immagini++;
                std[i] += ((float) image[j][i] - mean[i]) * (image[j][i] - mean[i]);
            }
        }
        if (immagini > 0)
            std[i] = sqrt(std[i] / immagini);
    }
}
void filterPixelsCPU(u_int16_t *mean, float *std, u_int16_t **image, int k, int numImages, int npixels) {
    for (int i = 0; i < npixels; i++) {
        for (int j = 0; j < numImages; j++) {
            if (image[j][i] > mean[i] + k * std[i] || image[j][i] < mean[i] - k * std[i]) {
                image[j][i] = 0;
            }
        }
    }
}

#endif // HOST_ALFA_SIGMA_H