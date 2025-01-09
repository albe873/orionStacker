#ifndef CUDA_DEVICE_ALFA_SIGMA_H
#define CUDA_DEVICE_ALFA_SIGMA_H


// calcolo media di tutte le immagini escludendo i pixel con valore 0
__device__ inline void computeMean2(u_int16_t **image, u_int16_t *mean, u_int64_t idx1, u_int64_t idx2, u_int16_t numImages) {
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

__device__ inline void computePartialMean2(u_int16_t **image, u_int16_t* mean1, u_int16_t* mean2, u_int64_t idx1, u_int64_t idx2, u_int16_t numImages) {
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
__device__ inline void computeStdDev2(float *std1, float *std2, u_int16_t mean1, u_int16_t mean2, u_int16_t **image, u_int64_t idx1, u_int64_t idx2, u_int16_t numImages) {
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
__device__ inline void filterPixels2(u_int16_t mean1, float std1, u_int16_t mean2, float std2, u_int16_t **image, u_int64_t idx1, u_int64_t idx2, float k, u_int16_t numImages) {
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

__global__ void compute_alfa_sigma2(u_int16_t **image, u_int16_t *mean, u_int16_t numImages, u_int64_t npixels, float k, u_int16_t s) {
    u_int64_t idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    idx1 *= 2;
    u_int64_t idx2 = idx1 + 1;

    float std1, std2;
    u_int16_t part_mean1, part_mean2;
    
    if (idx2 < npixels) {
        for (u_int16_t i = 0; i < s; i++) {
            computePartialMean2(image, &part_mean1, &part_mean2, idx1, idx2, numImages);
            computeStdDev2(&std1, &std2, part_mean1, part_mean2, image, idx1, idx2, numImages);
            filterPixels2(part_mean1, std1, part_mean2, std2, image, idx1, idx2, k, numImages);
        }
        computeMean2(image, mean, idx1, idx2, numImages);
    }

}

#endif // CUDA_DEVICE_ALFA_SIGMA_H