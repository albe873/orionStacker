#include "cuda_helper.hh"

// -------------- uint16_t version --------------

// funzioni con versioni per singolo pixel e per 2 pixel
// calcolo 2 pixel per thread per migliorare l'efficacia della cache
// (linee da 128 byte: 2 (pixel) * 2 (byte per pixel) * 32 (warp size) = 128 byte)

// calcolo media di tutte le immagini escludendo i pixel con valore 0,
// output del valore calcolato nell'array finale

__device__ inline void stk_mean_2(uint16_t *img_all, uint16_t *mean, 
                                   uint64_t idx1, uint64_t idx2, 
                                   int n_img, uint64_t npixels) {
    uint16_t count1 = 0, count2 = 0;
    uint32_t acc1 = 0, acc2 = 0;  // max 2^16 n_img 
    for (int i = 0; i < n_img; i++) {
        uint64_t base = i * npixels;
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

__device__ inline void stk_mean(uint16_t *img_all, uint16_t *mean, 
                                 uint64_t idx, int n_img, uint64_t npixels) {
    uint16_t count = 0;
    uint32_t acc = 0;
    for (int i = 0; i < n_img; i++) {
        uint64_t base = (uint64_t)i * npixels;
        uint16_t val = img_all[base + idx];
        if (val > 0) {
            count++;
            acc += val;
        }
    }
    mean[idx] = (count > 0) ? acc / count : 0;
}

__device__ inline void partial_mean_2(uint16_t *img_all, 
                                      uint16_t* mean1, uint16_t* mean2,
                                      uint64_t idx1, uint64_t idx2,
                                      int n_img, uint64_t npixels) {
    uint32_t acc1 = 0, acc2 = 0;
    uint16_t count1 = 0, count2 = 0;
    for (int i = 0; i < n_img; i++) {
        uint64_t base = i * npixels;
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

__device__ inline void partial_mean(uint16_t *img_all, uint16_t* mean, uint64_t idx, int n_img, uint64_t npixels) {
    uint32_t acc = 0;
    uint16_t count = 0;
    for (int i = 0; i < n_img; i++) {
        uint64_t base = i * npixels;
        auto val = img_all[base + idx];
        if (val > 0) {
            acc += val;
            count++;
        }
    }
    *mean = (count > 0) ? acc / count : 0;
}

__device__ inline void var_imgs_2(float *var1, float *var2, 
                                  uint16_t mean1, uint16_t mean2,
                                  uint16_t *img_all,
                                  uint64_t idx1, uint64_t idx2,
                                  int n_img, uint64_t npixels) {
    uint16_t count1 = 0, count2 = 0;
    *var1 = 0.0f;
    *var2 = 0.0f;
    for (int i = 0; i < n_img; i++) {
        uint64_t base = i * npixels;
        auto val1 = img_all[base + idx1];
        auto val2 = img_all[base + idx2];
        if (val1 > 0) {
            count1++;
            *var1 += ((float)val1 - mean1) * ((float)val1 - mean1);
        }
        if (val2 > 0) {
            count2++;
            *var2 += ((float)val2 - mean2) * ((float)val2 - mean2);
        }
    }
    *var1 = (count1 > 2) ? (*var1 / (count1-1)) : 0.0f;
    *var2 = (count2 > 2) ? (*var2 / (count2-1)) : 0.0f;
}

__device__ inline void var_imgs(float *var, uint16_t mean, uint16_t *img_all,
                                uint64_t idx, int n_img, uint64_t npixels) {
    uint16_t count = 0;
    *var = 0.0f;
    for (int i = 0; i < n_img; i++) {
        uint64_t base = i * npixels;
        auto val = img_all[base + idx];
        if (val > 0) {
            count++;
            *var += ((float)val - mean) * ((float)val - mean);
        }
    }
    *var = (count > 2) ? (*var / (count-1)) : 0.0f;
}

__device__ inline void filter_pixels_2(uint16_t mean1, float var1, 
                                       uint16_t mean2, float var2,
                                       uint16_t *img_all,
                                       uint64_t idx1, uint64_t idx2,
                                       float k, 
                                       int n_img, uint64_t npixels) {
    float s1 = k * k * var1;
    float s2 = k * k * var2;
    for (int i = 0; i < n_img; i++) {
        uint64_t base = i * npixels;
        auto val1 = img_all[base + idx1];
        auto val2 = img_all[base + idx2];

        if (val1 > 0) {
            float d1 = (float)val1 - (float)mean1;
            if (d1 * d1 > s1)
                img_all[base + idx1] = 0;
        }

        if (val2 > 0) {
            float d2 = (float)val2 - (float)mean2;
            if (d2 * d2 > s2)
                img_all[base + idx2] = 0;
        }
    }
}

__device__ inline void filter_pixels(uint16_t mean, float var, uint16_t *img_all,
                                     uint64_t idx, float k, int n_img, uint64_t npixels) {
    float s = k * k * var;
    for (int i = 0; i < n_img; i++) {
        uint64_t base = i * npixels;
        uint16_t val = img_all[base + idx];
        if (val > 0) {
            float d = (float)val - (float)mean;
            if (d * d > s)
                img_all[base + idx] = 0;
        }
    }
}

__global__ void kernel_alfa_sigma(uint16_t *img_all, uint16_t *mean, 
                                  int n_img,  uint64_t npixels, 
                                  float k, int it) {
    const uint64_t idx1 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const uint64_t idx2 = idx1 + 1;

    float var1, var2;
    uint16_t part_mean1, part_mean2;
    
    if (idx2 < npixels) {
        for (int i = 0; i < it; i++) {
            partial_mean_2(img_all, &part_mean1, &part_mean2, idx1, idx2, n_img, npixels);
            var_imgs_2(&var1, &var2, part_mean1, part_mean2, img_all, idx1, idx2, n_img, npixels);
            filter_pixels_2(part_mean1, var1, part_mean2, var2, img_all, idx1, idx2, k, n_img, npixels);
        }
        stk_mean_2(img_all, mean, idx1, idx2, n_img, npixels);
    }

    if (idx2 == npixels) { // caso dispari
        for (int i = 0; i < it; i++) {
            partial_mean(img_all, &part_mean1, idx1, n_img, npixels);
            var_imgs(&var1, part_mean1, img_all, idx1, n_img, npixels);
            filter_pixels(part_mean1, var1, img_all, idx1, k, n_img, npixels);
        }
        stk_mean(img_all, mean, idx1, n_img, npixels);
    }
}

void alfa_sigma_gpu(uint16_t *img_all, uint16_t *mean, 
                    int n_img, const uint64_t npixels, 
                    const float k, const int it) {
    dim3 block_size(512);
    dim3 grid_size = (npixels / 2 + block_size.x - 1) / block_size.x;

    kernel_alfa_sigma<<<grid_size, block_size>>>(img_all, mean, n_img, npixels, k, it);
    CHECK(cudaDeviceSynchronize());
}

// =============================================================================
// simple_winsorized_sigma_clipping — GPU
// =============================================================================

__device__ inline void clip_oob_pixels_2(uint16_t mean1, float std1,
                                         uint16_t mean2, float std2,
                                         uint16_t *img_all,
                                         uint64_t idx1, uint64_t idx2,
                                         float k_low, float k_high,
                                         int n_img, uint64_t npixels) {
    float f_lb1 = mean1 - k_low  * std1;
    float f_ub1 = mean1 + k_high * std1;
    float f_lb2 = mean2 - k_low  * std2;
    float f_ub2 = mean2 + k_high * std2;

    uint16_t lb1 = f_lb1 > 0 ? f_lb1 : 0;
    uint16_t ub1 = f_ub1 < 65535 ? f_ub1 : 65535;
    uint16_t lb2 = f_lb2 > 0 ? f_lb2 : 0;
    uint16_t ub2 = f_ub2 < 65535 ? f_ub2 : 65535;

    for (int i = 0; i < n_img; i++) {
        uint64_t base = i * npixels;
        auto val1 = img_all[base + idx1];
        auto val2 = img_all[base + idx2];

        if (val1 > 0) {
            if (val1 > ub1)
                img_all[base + idx1] = ub1;
            else if (val1 < lb1)
                img_all[base + idx1] = lb1;
        }

        if (val2 > 0) {
            if (val2 > ub2)
                img_all[base + idx2] = ub2;
            else if (val2 < lb2)
                img_all[base + idx2] = lb2;
        }
    }
}

__device__ inline void clip_oob_pixels(uint16_t mean, float std,
                                       uint16_t *img_all,
                                       uint64_t idx,
                                       float k_low, float k_high,
                                       int n_img, uint64_t npixels) {
    float f_lb = mean - k_low  * std;
    float f_ub = mean + k_high * std;
    uint16_t lb = f_lb > 0 ? f_lb : 0;
    uint16_t ub = f_ub < 65535 ? f_ub : 65535;
    for (int i = 0; i < n_img; i++) {
        uint64_t base = i * npixels;
        auto val = img_all[base + idx];
        if (val > 0) {
            if (val > ub)
                img_all[base + idx] = ub;
            else if (val < lb)
                img_all[base + idx] = lb;
        }
    }
}

__global__ void kernel_simple_winsorized_sigma_clipping(uint16_t *img_all, uint16_t *mean,
                                                        int n_img, uint64_t npixels,
                                                        float k_low, float k_high,
                                                        int iterations) {
    const uint64_t idx1 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const uint64_t idx2 = idx1 + 1;

    float var1, var2, std1, std2;
    uint16_t part_mean1, part_mean2;

    if (idx2 < npixels) {
        for (int it = 0; it < iterations; it++) {
            partial_mean_2(img_all, &part_mean1, &part_mean2, idx1, idx2, n_img, npixels);
            var_imgs_2(&var1, &var2, part_mean1, part_mean2, img_all, idx1, idx2, n_img, npixels);
            std1 = sqrtf(var1);
            std2 = sqrtf(var2);
            clip_oob_pixels_2(part_mean1, std1, part_mean2, std2, img_all, idx1, idx2,
                              k_low, k_high, n_img, npixels);
        }
        stk_mean_2(img_all, mean, idx1, idx2, n_img, npixels);
    }

    if (idx2 == npixels) { // odd case
        for (int it = 0; it < iterations; it++) {
            partial_mean(img_all, &part_mean1, idx1, n_img, npixels);
            var_imgs(&var1, part_mean1, img_all, idx1, n_img, npixels);
            std1 = sqrt(var1);
            clip_oob_pixels(part_mean1, var1, img_all, idx1, k_low, k_high, n_img, npixels);
        }
        stk_mean(img_all, mean, idx1, n_img, npixels);
    }
}

void simple_winsorized_sigma_clipping_gpu(uint16_t *img_all, uint16_t *mean,
                                           int numImages, int npixels,
                                           float kappa_low, float kappa_high,
                                           int iterations) {
    dim3 block_size(512);
    dim3 grid_size = ((uint64_t)npixels / 2 + block_size.x - 1) / block_size.x;

    kernel_simple_winsorized_sigma_clipping<<<grid_size, block_size>>>(
        img_all, mean, numImages, (uint64_t)npixels, kappa_low, kappa_high, iterations);
    CHECK(cudaDeviceSynchronize());
}


__device__ inline void set_to_mean_oob_pixels_2(uint16_t mean1, float std1,
                                                 uint16_t mean2, float std2,
                                                 uint16_t *img_all,
                                                 uint64_t idx1, uint64_t idx2,
                                                 float k_low, float k_high,
                                                 int n_img, uint64_t npixels) {
    float f_lb1 = mean1 - k_low  * std1;
    float f_ub1 = mean1 + k_high * std1;
    float f_lb2 = mean2 - k_low  * std2;
    float f_ub2 = mean2 + k_high * std2;

    uint16_t lb1 = f_lb1 > 0 ? f_lb1 : 0;
    uint16_t ub1 = f_ub1 < 65535 ? f_ub1 : 65535;
    uint16_t lb2 = f_lb2 > 0 ? f_lb2 : 0;
    uint16_t ub2 = f_ub2 < 65535 ? f_ub2 : 65535;

    for (int i = 0; i < n_img; i++) {
        uint64_t base = i * npixels;
        auto val1 = img_all[base + idx1];
        auto val2 = img_all[base + idx2];

        if (val1 > 0 && (val1 < lb1 || val1 > ub1))
            img_all[base + idx1] = mean1;
        if (val2 > 0 && (val2 < lb2 || val2 > ub2))
            img_all[base + idx2] = mean2;
    }
}

// Single-pixel version: sigma is STANDARD DEVIATION (from MAD)
__device__ inline void set_to_mean_oob_pixels(uint16_t mean, float std,
                                               uint16_t *img_all,
                                               uint64_t idx,
                                               float k_low, float k_high,
                                               int n_img, uint64_t npixels) {
    float f_lb = mean - k_low  * std;
    float f_ub = mean + k_high * std;
    uint16_t lb = f_lb > 0 ? f_lb : 0;
    uint16_t ub = f_ub < 65535 ? f_ub : 65535;

    for (int i = 0; i < n_img; i++) {
        uint64_t base = i * npixels;
        auto val = img_all[base + idx];
        
        if (val > 0 && (val < lb || val > ub))
            img_all[base + idx] = mean;
    }
}

// Count non-zero values at a single pixel position across all images
__device__ inline int count_nonzero_at(uint16_t *img_all, uint64_t idx,
                                        int n_img, uint64_t npixels) {
    int count = 0;
    for (int j = 0; j < n_img; j++)
        if (img_all[npixels * j + idx] > 0)
            count++;
    return count;
}

// Find the k-th smallest non-zero value (0-indexed) via binary search on [1, 65535].
__device__ inline uint16_t kth_nonzero_value(uint16_t *img_all, uint64_t idx,
                                              int n_img, uint64_t npixels,
                                              int total_count, int k) {
    int lo = 1, hi = 65535;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        int le = 0;
        for (int j = 0; j < n_img; j++) {
            uint16_t v = img_all[npixels * j + idx];
            if (v > 0 && (int)v <= mid)
                le++;
        }
        if (le > k)
            hi = mid;
        else
            lo = mid + 1;
    }
    return (uint16_t)lo;
}

// Find the k-th smallest absolute deviation |v - median| (0-indexed) via binary search.
__device__ inline uint16_t kth_deviation(uint16_t *img_all, uint64_t idx,
                                          int n_img, uint64_t npixels,
                                          uint16_t median, int total_count, int k) {
    int lo = 0, hi = 65535;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        int le = 0;
        for (int j = 0; j < n_img; j++) {
            uint16_t v = img_all[npixels * j + idx];
            if (v > 0) {
                int diff = (int)v - (int)median;
                if (diff < 0)
                    diff = -diff;
                if (diff <= mid)
                    le++;
            }
        }
        if (le > k)
            hi = mid;
        else
            lo = mid + 1;
    }
    return (uint16_t)lo;
}

// Computes median and MAD-based robust sigma for a single pixel position.
// Uses binary-search quickselect on global memory — O(1) extra memory, no limit on n_img.
__device__ inline void median_mad_sigma(uint16_t *img_all, uint64_t idx,
                                         int n_img, uint64_t npixels,
                                         uint16_t *median_out, float *sigma_out) {
    int count = count_nonzero_at(img_all, idx, n_img, npixels);

    if (count == 0) {
        *median_out = 0;
        *sigma_out = 0.0f;
        return;
    }

    uint16_t med = kth_nonzero_value(img_all, idx, n_img, npixels, count, count / 2);
    *median_out = med;

    if (count < 2) {
        *sigma_out = 0.0f;
        return;
    }

    uint16_t mad = kth_deviation(img_all, idx, n_img, npixels, med, count, count / 2);
    *sigma_out = 1.4826f * (float)mad;
}

// Two-pixel version of median+MAD
__device__ inline void median_mad_sigma_2(uint16_t *img_all,
                                           uint64_t idx1, uint64_t idx2,
                                           int n_img, uint64_t npixels,
                                           uint16_t *median1, float *sigma1,
                                           uint16_t *median2, float *sigma2) {
    median_mad_sigma(img_all, idx1, n_img, npixels, median1, sigma1);
    median_mad_sigma(img_all, idx2, n_img, npixels, median2, sigma2);
}

__global__ void kernel_winsorized_sigma_clipping(uint16_t *img_all, uint16_t *mean,
                                                  int n_img, uint64_t npixels,
                                                  float k_low, float k_high,
                                                  float k1_low, float k2_high,
                                                  float conv_tolerance) {
    const uint64_t idx1 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const uint64_t idx2 = idx1 + 1;
    const int max_iterations = 100;

    float var1, var2, std1, std2, prev_change;
    uint16_t part_mean1, part_mean2;
    uint16_t prev_mean1, prev_mean2;

    if (idx2 < npixels) {
        // Phase 1: median + MAD (robust to outliers)
        median_mad_sigma_2(img_all, idx1, idx2, n_img, npixels,
                           &part_mean1, &std1, &part_mean2, &std2);
        mean[idx1] = part_mean1;
        mean[idx2] = part_mean2;
        set_to_mean_oob_pixels_2(part_mean1, std1, part_mean2, std2, img_all, idx1, idx2,
                                  k_low, k_high, n_img, npixels);

        // Phase 2: iterate with mean + classical std
        for (int iter = 0; iter < max_iterations; iter++) {
            prev_mean1 = mean[idx1];
            prev_mean2 = mean[idx2];

            partial_mean_2(img_all, &part_mean1, &part_mean2, idx1, idx2, n_img, npixels);
            var_imgs_2(&var1, &var2, part_mean1, part_mean2, img_all, idx1, idx2, n_img, npixels);
            std1 = sqrtf(var1);
            std2 = sqrtf(var2);
            clip_oob_pixels_2(part_mean1, std1, part_mean2, std2, img_all, idx1, idx2,
                              k1_low, k2_high, n_img, npixels);

            mean[idx1] = part_mean1;
            mean[idx2] = part_mean2;

            prev_change = fabsf((float)part_mean1 - (float)prev_mean1) +
                          fabsf((float)part_mean2 - (float)prev_mean2);

            if (prev_change / 2.0f < conv_tolerance)
                break;
        }

        // Final mean
        stk_mean_2(img_all, mean, idx1, idx2, n_img, npixels);
    }

    if (idx2 == npixels) { // odd case
        // Phase 1
        median_mad_sigma(img_all, idx1, n_img, npixels, &part_mean1, &std1);
        mean[idx1] = part_mean1;
        set_to_mean_oob_pixels(part_mean1, std1, img_all, idx1, k_low, k_high, n_img, npixels);

        // Phase 2
        for (int iter = 0; iter < max_iterations; iter++) {
            prev_mean1 = mean[idx1];

            partial_mean(img_all, &part_mean1, idx1, n_img, npixels);
            var_imgs(&std1, part_mean1, img_all, idx1, n_img, npixels);
            clip_oob_pixels(part_mean1, std1, img_all, idx1, k1_low, k2_high, n_img, npixels);

            mean[idx1] = part_mean1;

            prev_change = fabsf((float)part_mean1 - (float)prev_mean1);
            if (prev_change < conv_tolerance)
                break;
        }

        stk_mean(img_all, mean, idx1, n_img, npixels);
    }
}

void winsorized_sigma_clipping_gpu(uint16_t *img_all, uint16_t *mean,
                                    int n_img, int npixels,
                                    float k_low, float k_high,
                                    float k1_low, float k2_high,
                                    float conv_tolerance) {
    dim3 block_size(512);
    dim3 grid_size = ((uint64_t)npixels / 2 + block_size.x - 1) / block_size.x;

    kernel_winsorized_sigma_clipping<<<grid_size, block_size>>>(
        img_all, mean, n_img, (uint64_t)npixels,
        k_low, k_high, k1_low, k2_high, conv_tolerance);
    CHECK(cudaDeviceSynchronize());
}