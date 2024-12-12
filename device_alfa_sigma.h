#ifndef CUDA_DEVICE_ALFA_SIGMA_H
#define CUDA_DEVICE_ALFA_SIGMA_H


// calcolo media di tutte le immagini escludendo i pixel con valore 0
__device__ inline void computeMean(u_int16_t **image, u_int16_t *mean, int idx, int numImages, int npixels) {
    u_int16_t immagini = 0;
    u_int32_t acc = 0;
    for (int i = 0; i < numImages; i++) {
        if (image[i][idx] > 0) {
            immagini++;
            acc += image[i][idx];
        }
    }
    mean[idx] = (immagini > 0) ? acc / immagini : 0;
}
__device__ inline void computePartialMean(u_int16_t **image, u_int16_t* mean, int idx, int numImages, int npixels) {
    u_int16_t immagini = 0;
    u_int32_t acc = 0;
    for (int i = 0; i < numImages; i++) {
        if (image[i][idx] > 0) {
            immagini++;
            acc += image[i][idx];
        }
    }
    *mean = (immagini > 0) ? acc / immagini : 0;
}


// calcolo deviazione standard
__device__ inline void computeStdDev(float *std, u_int16_t mean, u_int16_t **image, int idx, int numImages, int npixels) {
    u_int16_t immagini = 0;
    *std = 0.0f;
    for (int i = 0; i < numImages; i++) {
        if (image[i][idx] > 0) {
            immagini++;
            *std += ((float) image[i][idx] - mean) * (image[i][idx] - mean);
        }
    }
    *std = immagini > 0 ? sqrt(*std / immagini) : 0.0f;
}

// filtro i pixel con valore fuori dal range, mettendoli a 0
__device__ inline void filterPixels(u_int16_t *mean, float std, u_int16_t **image, int idx, int k, int numImages, int npixels) {
    for (int i = 0; i < numImages; i++) {
        if (image[i][idx] > mean[idx] + (k * std) || image[i][idx] < mean[idx] - (k * std)) {
            image[i][idx] = 0;
        }
    }
}

__global__ void compute_alfa_sigma(u_int16_t **image, u_int16_t *mean, int numImages, int npixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float std;
    u_int16_t part_mean;
    if (idx < npixels) {
        for (int i = 0; i < 5; i++) {
            computePartialMean(image, &part_mean, idx, numImages, npixels);
            computeStdDev(&std, part_mean, image, idx, numImages, npixels);
            filterPixels(mean, std, image, idx, 3, numImages, npixels);
        }
        computeMean(image, mean, idx, numImages, npixels);
    }
}

#endif // CUDA_DEVICE_ALFA_SIGMA_H