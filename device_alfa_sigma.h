#ifndef CUDA_DEVICE_ALFA_SIGMA_H
#define CUDA_DEVICE_ALFA_SIGMA_H

// accumulo dei valori di ogni pixel delle immagini con SATURAZIONE
__global__ void accumulatePixels(u_int32_t *acc_d, u_int16_t *d_image, int npixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < npixels) {
        acc_d[idx] += d_image[idx];
    }
}

// calcolo media di tutte le immagini escludendo i pixel con valore 0
__global__ void computeMeanAdv(u_int16_t **image, u_int16_t *mean, int numImages, int npixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    u_int16_t immagini = 0;
    u_int32_t acc = 0;
    if (idx < npixels) {
        for (int i = 0; i < numImages; i++) {
            if (image[i][idx] > 0) {
                immagini++;
                acc += image[i][idx];
            }
        }
        if (immagini > 0)
            mean[idx] = acc / immagini;
        else
            mean[idx] = 0;
    }
}

// calcolo deviazione standard
__global__ void computeStdDev(float *std, u_int16_t *mean, u_int16_t **image, int numImages, int npixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    u_int16_t immagini = 0;
    if (idx < npixels) {
        std[idx] = 0.0f;
        for (int i = 0; i < numImages; i++) {
            if (image[i][idx] > 0) {
                immagini++;
                std[idx] += ((float) image[i][idx] - mean[idx]) * (image[i][idx] - mean[idx]);
            }
        }
        if (immagini > 0)
            std[idx] = sqrt(std[idx] / immagini);
        else 
            std[idx] = 0.0f;
    }
}

// filtro i pixel con valore fuori dal range, mettendoli a 0
__global__ void filterPixels(u_int16_t *mean, float *std, u_int16_t **image, int k, int numImages, int npixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < npixels) {
        for (int i = 0; i < numImages; i++) {
            if (image[i][idx] > mean[idx] + k * std[idx] || image[i][idx] < mean[idx] - k * std[idx]) {
                image[i][idx] = 0;
            }
        }
    }
}

#endif // CUDA_DEVICE_ALFA_SIGMA_H