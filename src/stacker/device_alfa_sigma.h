#ifndef CUDA_DEVICE_ALFA_SIGMA_H
#define CUDA_DEVICE_ALFA_SIGMA_H


// -------------- uint16_t version --------------

// funzioni con versioni per singolo pixel e per 2 pixel
// calcolo 2 pixel per thread per migliorare l'efficacia della cache
// (linee da 128 byte: 2 (pixel) * 2 (byte per pixel) * 32 (warp size) = 128 byte)

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

__device__ inline void computeMean_uint16(u_int16_t **image, u_int16_t *mean, u_int64_t idx, u_int16_t numImages) {
    u_int16_t count = 0;
    u_int32_t acc = 0;
    for (int i = 0; i < numImages; i++) {
        u_int16_t val = image[i][idx];
        if (val > 0) {
            count++;
            acc += val;
        }
    }
    mean[idx] = (count > 0) ? acc / count : 0;
}

// calcolo media di tutte le immagini escludendo i pixel con valore 0,
// output del valore calcolato in variabile passata per riferimento
__device__ inline void computePartialMean2_uint16(u_int16_t **image, u_int16_t* mean1, u_int16_t* mean2, u_int64_t idx1, u_int64_t idx2, u_int16_t numImages) {
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

__device__ inline void computePartialMean_uint16(u_int16_t **image, u_int16_t* mean, u_int64_t idx, u_int16_t numImages) {
    u_int32_t acc = 0;
    u_int16_t count = 0;
    for (int i = 0; i < numImages; i++) {
        u_int16_t val = image[i][idx];
        if (val > 0) {
            acc += val;
            count++;
        }
    }
    *mean = (count > 0) ? acc / count : 0;
}

// calcolo deviazione standard
__device__ inline void computeStdDev2_uint16(float *std1, float *std2, u_int16_t mean1, u_int16_t mean2, u_int16_t **image, u_int64_t idx1, u_int64_t idx2, u_int16_t numImages) {
    u_int16_t count1 = 0, count2 = 0;
    *std1 = 0.0f;
    *std2 = 0.0f;
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
    *std1 = (count1 > 1) ? sqrtf(*std1 / count1) : 0.0f;
    *std2 = (count2 > 1) ? sqrtf(*std2 / count2) : 0.0f;
}

__device__ inline void computeStdDev_uint16(float *std, u_int16_t mean, u_int16_t **image, u_int64_t idx, u_int16_t numImages) {
    u_int16_t count = 0;
    *std = 0.0f;
    for (int i = 0; i < numImages; i++) {
        u_int16_t val = image[i][idx];
        if (val > 0) {
            count++;
            *std += ((float)val - mean) * ((float)val - mean);
        }
    }
    *std = (count > 1) ? sqrtf(*std / count) : 0.0f;
}

// filtro i pixel con valore fuori dal range, mettendoli a 0
__device__ inline void filterPixels2_uint16(u_int16_t mean1, float std1, u_int16_t mean2, float std2, u_int16_t **image, u_int64_t idx1, u_int64_t idx2, float k, u_int16_t numImages) {
    for (int i = 0; i < numImages; i++) {
        u_int16_t val1 = image[i][idx1];
        u_int16_t val2 = image[i][idx2];

        if (val1 > 0 && (val1 > mean1 + k * std1 || val1 < mean1 - k * std1))
            image[i][idx1] = 0;

        if (val2 > 0 && (val2 > mean2 + k * std2 || val2 < mean2 - k * std2))
            image[i][idx2] = 0;
    }
}

__device__ inline void filterPixels_uint16(u_int16_t mean, float std, u_int16_t **image, u_int64_t idx, float k, u_int16_t numImages) {
    for (int i = 0; i < numImages; i++) {
        u_int16_t val = image[i][idx];
        if (val > 0 && (val > mean + k * std || val < mean - k * std))
            image[i][idx] = 0;
    }
}


__global__ void compute_alfa_sigma_uint16(u_int16_t **image, u_int16_t *mean, const u_int16_t numImages, const u_int64_t npixels, const float k, const u_int16_t s) {
    const u_int64_t idx1 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const u_int64_t idx2 = idx1 + 1;

    float std1, std2;
    u_int16_t part_mean1, part_mean2;
    
    if (idx2 < npixels) {
        for (u_int16_t i = 0; i < s; i++) {
            computePartialMean2_uint16(image, &part_mean1, &part_mean2, idx1, idx2, numImages);
            computeStdDev2_uint16(&std1, &std2, part_mean1, part_mean2, image, idx1, idx2, numImages);
            filterPixels2_uint16(part_mean1, std1, part_mean2, std2, image, idx1, idx2, k, numImages);
        }
        computeMean2_uint16(image, mean, idx1, idx2, numImages);
    }

    if (idx2 == npixels) { // caso dispari, ultimo pixel idx1
        for (u_int16_t i = 0; i < s; i++) {
            computePartialMean_uint16(image, &part_mean1, idx1, numImages);
            computeStdDev_uint16(&std1, part_mean1, image, idx1, numImages);
            filterPixels_uint16(part_mean1, std1, image, idx1, k, numImages);
        }
        computeMean_uint16(image, mean, idx1, numImages);
    }
}

// -------------- uint8_t version --------------

// funzioni con versioni per singolo pixel e per 2 pixel
// calcolo 4 pixel per thread per migliorare l'efficacia della cache
// (linee da 128 byte: 4 (pixel) * 1 (byte per pixel) * 32 (warp size) = 128 byte)

// calcolo media di tutte le immagini escludendo i pixel con valore 0,
// output del valore calcolato nell'array finale
__device__ inline void computeMean4_uint8(u_int8_t **image, u_int8_t *mean, 
                                          u_int64_t idx1, u_int64_t idx2, u_int64_t idx3, u_int64_t idx4, 
                                          u_int16_t numImages) {
    u_int16_t count1 = 0, count2 = 0, count3 = 0, count4 = 0;
    u_int32_t acc1 = 0, acc2 = 0, acc3 = 0, acc4 = 0;
    for (int i = 0; i < numImages; i++) {
        u_int16_t val1 = image[i][idx1];
        u_int16_t val2 = image[i][idx2];
        u_int16_t val3 = image[i][idx3];
        u_int16_t val4 = image[i][idx4];
        if (val1 > 0) {
            count1++;
            acc1 += val1;
        }
        if (val2 > 0) {
            count2++;
            acc2 += val2;
        }
        if (val3 > 0) {
            count3++;
            acc3 += val3;
        }
        if (val4 > 0) {
            count4++;
            acc4 += val4;
        }
    }
    mean[idx1] = (count1 > 0) ? acc1 / count1 : 0;
    mean[idx2] = (count2 > 0) ? acc2 / count2 : 0;
    mean[idx3] = (count3 > 0) ? acc3 / count3 : 0;
    mean[idx4] = (count4 > 0) ? acc4 / count4 : 0;
}

__device__ inline void computeMean_uint8(u_int8_t **image, u_int8_t *mean, u_int64_t idx, u_int16_t numImages) {
    u_int16_t count = 0;
    u_int32_t acc = 0;
    for (int i = 0; i < numImages; i++) {
        u_int8_t val = image[i][idx];
        if (val > 0) {
            count++;
            acc += val;
        }
    }
    mean[idx] = (count > 0) ? acc / count : 0;
}

// calcolo media di tutte le immagini escludendo i pixel con valore 0,
// output del valore calcolato in variabile passata per riferimento
__device__ inline void computePartialMean4_uint8(u_int8_t **image, 
                                                 u_int8_t* mean1, u_int8_t* mean2, u_int8_t* mean3, u_int8_t* mean4, 
                                                 u_int64_t idx1, u_int64_t idx2, u_int64_t idx3, u_int64_t idx4, 
                                                 u_int16_t numImages) {
    u_int32_t acc1 = 0, acc2 = 0, acc3 = 0, acc4 = 0;
    u_int16_t count1 = 0, count2 = 0, count3 = 0, count4 = 0;
    for (int i = 0; i < numImages; i++) {
        u_int16_t val1 = image[i][idx1];
        u_int16_t val2 = image[i][idx2];
        u_int16_t val3 = image[i][idx3];
        u_int16_t val4 = image[i][idx4];
        if (val1 > 0) {
            acc1 += val1;
            count1++;
        }
        if (val2 > 0) {
            acc2 += val2;
            count2++;
        }
        if (val3 > 0) {
            acc3 += val3;
            count3++;
        }
        if (val4 > 0) {
            acc4 += val4;
            count4++;
        }
    }
    *mean1 = (count1 > 0) ? acc1 / count1 : 0;
    *mean2 = (count2 > 0) ? acc2 / count2 : 0;
    *mean3 = (count3 > 0) ? acc3 / count3 : 0;
    *mean4 = (count4 > 0) ? acc4 / count4 : 0;
}

__device__ inline void computePartialMean_uint8(u_int8_t **image, u_int8_t* mean, u_int64_t idx, u_int16_t numImages) {
    u_int32_t acc = 0;
    u_int16_t count = 0;
    for (int i = 0; i < numImages; i++) {
        u_int8_t val = image[i][idx];
        if (val > 0) {
            acc += val;
            count++;
        }
    }
    *mean = (count > 0) ? acc / count : 0;
}

// calcolo deviazione standard
__device__ inline void computeStdDev4_uint8(float *std1, float *std2, float *std3, float *std4, 
                                            u_int8_t mean1, u_int8_t mean2, u_int8_t mean3, u_int8_t mean4, 
                                            u_int8_t **image, 
                                            u_int64_t idx1, u_int64_t idx2, u_int64_t idx3, u_int64_t idx4, 
                                            u_int16_t numImages) {
    u_int16_t count1 = 0, count2 = 0, count3 = 0, count4 = 0;
    *std1 = 0.0f;
    *std2 = 0.0f;
    *std3 = 0.0f;
    *std4 = 0.0f;
    for (int i = 0; i < numImages; i++) {
        u_int16_t val1 = image[i][idx1];
        u_int16_t val2 = image[i][idx2];
        u_int16_t val3 = image[i][idx3];
        u_int16_t val4 = image[i][idx4];
        if (val1 > 0) {
            count1++;
            *std1 += ((float)val1 - mean1) * ((float)val1 - mean1);
        }
        if (val2 > 0) {
            count2++;
            *std2 += ((float)val2 - mean2) * ((float)val2 - mean2);
        }
        if (val3 > 0) {
            count3++;
            *std3 += ((float)val3 - mean3) * ((float)val3 - mean3);
        }
        if (val4 > 0) {
            count4++;
            *std4 += ((float)val4 - mean4) * ((float)val4 - mean4);
        }
    }
    *std1 = (count1 > 1) ? sqrtf(*std1 / count1) : 0.0f;
    *std2 = (count2 > 1) ? sqrtf(*std2 / count2) : 0.0f;
    *std3 = (count3 > 1) ? sqrtf(*std3 / count3) : 0.0f;
    *std4 = (count4 > 1) ? sqrtf(*std4 / count4) : 0.0f;
}

__device__ inline void computeStdDev_uint8(float *std, u_int8_t mean, u_int8_t **image, u_int64_t idx, u_int16_t numImages) {
    u_int16_t count = 0;
    *std = 0.0f;
    for (int i = 0; i < numImages; i++) {
        u_int8_t val = image[i][idx];
        if (val > 0) {
            count++;
            *std += ((float)val - mean) * ((float)val - mean);
        }
    }
    *std = (count > 1) ? sqrtf(*std / count) : 0.0f;
}

// filtro i pixel con valore fuori dal range, mettendoli a 0
__device__ inline void filterPixels4_uint8(u_int8_t mean1, float std1, u_int8_t mean2, float std2, u_int8_t mean3, float std3, u_int8_t mean4, float std4,
                                           u_int8_t **image,
                                           u_int64_t idx1, u_int64_t idx2, u_int64_t idx3, u_int64_t idx4, 
                                           float k, u_int16_t numImages) {
    for (int i = 0; i < numImages; i++) {
        u_int8_t val1 = image[i][idx1];
        u_int8_t val2 = image[i][idx2];
        u_int8_t val3 = image[i][idx3];
        u_int8_t val4 = image[i][idx4];

        if (val1 > 0 && (val1 > mean1 + k * std1 || val1 < mean1 - k * std1))
            image[i][idx1] = 0;

        if (val2 > 0 && (val2 > mean2 + k * std2 || val2 < mean2 - k * std2))
            image[i][idx2] = 0;

        if (val3 > 0 && (val3 > mean3 + k * std3 || val3 < mean3 - k * std3))
            image[i][idx3] = 0;
        
        if (val4 > 0 && (val4 > mean4 + k * std4 || val4 < mean4 - k * std4))
            image[i][idx4] = 0;
    }
}

__device__ inline void filterPixels_uint8(u_int8_t mean, float std, u_int8_t **image, u_int64_t idx, float k, u_int16_t numImages) {
    for (int i = 0; i < numImages; i++) {
        u_int8_t val = image[i][idx];
        if (val > 0 && (val > mean + k * std || val < mean - k * std))
            image[i][idx] = 0;
    }
}


__global__ void compute_alfa_sigma_uint8(u_int8_t **image, u_int8_t *mean, const u_int16_t numImages, const u_int64_t npixels, const float k, const u_int16_t s) {
    const u_int64_t idx1 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const u_int64_t idx2 = idx1 + 1;
    const u_int64_t idx3 = idx1 + 2;
    const u_int64_t idx4 = idx1 + 3;

    float std1, std2, std3, std4;
    u_int8_t part_mean1, part_mean2 , part_mean3, part_mean4;
    
    if (idx4 < npixels) {
        for (u_int16_t i = 0; i < s; i++) {
            computePartialMean4_uint8(image, &part_mean1, &part_mean2, &part_mean3, &part_mean4, idx1, idx2, idx3, idx4, numImages);
            computeStdDev4_uint8(&std1, &std2, &std3, &std4, part_mean1, part_mean2, part_mean3, part_mean4, image, idx1, idx2, idx3, idx4, numImages);
            filterPixels4_uint8(part_mean1, std1, part_mean2, std2, part_mean3, std3, part_mean4, std4, image, idx1, idx2, idx3, idx4, k, numImages);
        }
        computeMean4_uint8(image, mean, idx1, idx2, idx3, idx4, numImages);
        
    }   // caso non divisibile per 4, calcolo gli ultimi pixel
    else if (idx2 == npixels) { 
        for (u_int16_t i = 0; i < s; i++) {
            computePartialMean_uint8(image, &part_mean1, idx1, numImages);
            computeStdDev_uint8(&std1, part_mean1, image, idx1, numImages);
            filterPixels_uint8(part_mean1, std1, image, idx1, k, numImages);
        }
        computeMean_uint8(image, mean, idx1, numImages);
    }
    else if (idx3 == npixels) {
        for (u_int16_t i = 0; i < s; i++) {
            computePartialMean_uint8(image, &part_mean1, idx1, numImages);
            computeStdDev_uint8(&std1, part_mean1, image, idx1, numImages);
            filterPixels_uint8(part_mean1, std1, image, idx1, k, numImages);

            computePartialMean_uint8(image, &part_mean2, idx2, numImages);
            computeStdDev_uint8(&std2, part_mean2, image, idx2, numImages);
            filterPixels_uint8(part_mean2, std2, image, idx2, k, numImages);
        }
        computeMean_uint8(image, mean, idx1, numImages);
        computeMean_uint8(image, mean, idx2, numImages);
    }
    else if (idx4 == npixels) {
        for (u_int16_t i = 0; i < s; i++) {
            computePartialMean_uint8(image, &part_mean1, idx1, numImages);
            computeStdDev_uint8(&std1, part_mean1, image, idx1, numImages);
            filterPixels_uint8(part_mean1, std1, image, idx1, k, numImages);

            computePartialMean_uint8(image, &part_mean2, idx2, numImages);
            computeStdDev_uint8(&std2, part_mean2, image, idx2, numImages);
            filterPixels_uint8(part_mean2, std2, image, idx2, k, numImages);

            computePartialMean_uint8(image, &part_mean3, idx3, numImages);
            computeStdDev_uint8(&std3, part_mean3, image, idx3, numImages);
            filterPixels_uint8(part_mean3, std3, image, idx3, k, numImages);
        }
        computeMean_uint8(image, mean, idx1, numImages);
        computeMean_uint8(image, mean, idx2, numImages);
        computeMean_uint8(image, mean, idx3, numImages);
    }

}

#endif // CUDA_DEVICE_ALFA_SIGMA_H