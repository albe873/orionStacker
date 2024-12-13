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

__device__ inline void computeMean2(u_int16_t **image, u_int16_t *mean, int idx1, int idx2, int numImages, int npixels) {
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

__device__ inline void computePartialMean2(u_int16_t **image, u_int16_t* mean1, u_int16_t* mean2, int idx1, int idx2, int numImages) {
    u_int32_t acc1 = 0, acc2 = 0;
    u_int16_t count1 = 0, count2 = 0;
    for (int i = 0; i < numImages; i++) {
        u_int16_t val1 = image[i][idx1];
        u_int16_t val2 = image[i][idx2];
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

__device__ inline void computeStdDev2(float *std1, float *std2, u_int16_t mean1, u_int16_t mean2, u_int16_t **image, int idx1, int idx2, int numImages) {
    u_int16_t count1 = 0, count2 = 0;
    for (int i = 0; i < numImages; i++) {
        u_int16_t val1 = image[i][idx1];
        u_int16_t val2 = image[i][idx2];
        if (val1 > 0) {
            count1++;
            *std1 += ((float)val1 - mean1) * ((float)val1 - mean1);
        }
        if (val2 > 0) {
            count2++;
            *std2 += ((float)val2 - mean2) * ((float)val2 - mean2);
        }
    }
    *std1 = (count1 > 1) ? sqrtf(*std1 / (count1 - 1)) : 0.0f;
    *std2 = (count2 > 1) ? sqrtf(*std2 / (count2 - 1)) : 0.0f;
}

// filtro i pixel con valore fuori dal range, mettendoli a 0
__device__ inline void filterPixels(u_int16_t mean, float std, u_int16_t **image, int idx, int k, int numImages, int npixels) {
    for (int i = 0; i < numImages; i++) {
        if (image[i][idx] > mean + (k * std) || image[i][idx] < mean - (k * std)) {
            image[i][idx] = 0;
        }
    }
}

__device__ inline void filterPixels2(u_int16_t mean1, float std1, u_int16_t mean2, float std2, u_int16_t **image, int idx1, int idx2, float k, int numImages) {
    for (int i = 0; i < numImages; i++) {
        u_int16_t val1 = image[i][idx1];
        u_int16_t val2 = image[i][idx2];

        if (val1 > 0) {
            if (val1 > mean1 + k * std1 || val1 < mean1 - k * std1) {
                image[i][idx1] = 0;
            }
        }

        if (val2 > 0) {
            if (val2 > mean2 + k * std2 || val2 < mean2 - k * std2) {
                image[i][idx2] = 0;
            }
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
            filterPixels(part_mean, std, image, idx, 3, numImages, npixels);
        }
        computeMean(image, mean, idx, numImages, npixels);
    }
}

__global__ void compute_alfa_sigma2(u_int16_t **image, u_int16_t *mean, int numImages, int npixels) {
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    idx1 *= 2;
    int idx2 = idx1 + 1;

    float std1, std2;
    u_int16_t part_mean1, part_mean2;
    
    if (idx1 < npixels && idx2 < npixels) {
        for (int i = 0; i < 5; i++) {
            computePartialMean2(image, &part_mean1, &part_mean2, idx1, idx2, numImages);
            computeStdDev2(&std1, &std2, part_mean1, part_mean2, image, idx1, idx2, numImages);
            filterPixels2(part_mean1, std1, part_mean2, std2, image, idx1, idx2, 3, numImages);
        }
        computeMean2(image, mean, idx1, idx2, numImages, npixels);
    }

}

#endif // CUDA_DEVICE_ALFA_SIGMA_H