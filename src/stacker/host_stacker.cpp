#include <cstring>
#include <math.h>
#include <vector>
#include <algorithm>

#include "stacker.hh"

void inline mean_imgs(uint16_t* __restrict__ img_all, uint16_t* __restrict__ mean, uint64_t idx, int numImages, int npixels) {
    int count = 0;
    uint64_t acc = 0;
    for (int j = 0; j < numImages; j++) {
        uint64_t base = j * npixels;
        auto val = img_all[base + idx];
        if (val > 0) {
            count++;
            acc += val;
        }
    }
    mean[idx] = (count > 0) ? acc / count : 0;
}

void inline median_imgs(uint16_t* __restrict__ img_all, uint16_t* __restrict__ median, uint64_t idx, int n_img, int npixels) {
    std::vector<uint16_t> pixel_values(n_img);
    
    // Collect all values for this pixel position across all images
    int count = 0;
    for (int j = 0; j < n_img; j++) {
        auto val = img_all[npixels*j + idx];
        if (val > 0)
            pixel_values[count++] = val;
    }
    
    if (count == 0) {
        median[idx] = 0;
        return;
    }
    
    // Use nth_element to find median in O(n) average time - faster than sorting
    int mid = count / 2;
    std::nth_element(pixel_values.begin(), pixel_values.begin() + mid, pixel_values.begin() + count);
    
    median[idx] = pixel_values[mid];
    // for even sets I should make the mean of pixel_values[mid] and pixel_values[mid+1]
    // not done for semplicity
}

// Compute robust sigma using Median Absolute Deviation (MAD)
// MAD is insensitive to outliers, unlike classical std_dev
void inline mad_sigma(float &sigma, uint16_t* __restrict__ img_all, uint16_t* __restrict__ median, uint64_t idx, int n_img, int npixels) {
    std::vector<uint16_t> pixel_values(n_img);
    int count = 0;
    float med = (float)median[idx];
    
    // Collect absolute deviations from median
    for (int j = 0; j < n_img; j++) {
        auto val = img_all[npixels * j + idx];
        if (val > 0)
            pixel_values[count++] = (uint16_t)fabsf((float)val - med);
    }
    
    if (count < 2) {
        sigma = 0.0f;
        return;
    }
    
    // Find median of absolute deviations
    int mid = count / 2;
    std::nth_element(pixel_values.begin(), pixel_values.begin() + mid, pixel_values.begin() + count);
    
    // 1.4826 scales MAD to be equivalent to std_dev for normal distributions
    sigma = 1.4826f * (float)pixel_values[mid];
}


void inline var_imgs(float &var, uint16_t* __restrict__ mean, uint16_t* __restrict__ img_all, uint64_t idx, int numImages, int npixels) {
    int count = 0;
    var = 0.0f;
    for (int j = 0; j < numImages; j++) {
        uint64_t base = j * npixels;
        uint16_t val = img_all[base + idx];
        if (val > 0) {
            count++;
            var += ((float)val - mean[idx]) * ((float)val - mean[idx]);
        }
    }
    var = (count > 2) ? var / (count - 1) : 0.0f;
}

void inline filter_pixels(uint16_t* __restrict__ mean, float var, uint16_t* __restrict__ img_all, uint64_t idx, float k, int numImages, int npixels) {
    float s2 = k * k * var;
    for (int j = 0; j < numImages; j++) {
        uint64_t base = j * npixels;
        uint16_t val = img_all[base + idx];
        if (val > 0) {
            float d = (float)val - (float)mean[idx];
            if (d * d > s2)
                img_all[base + idx] = 0;
        }
    }
}

void alfa_sigma_cpu(
    uint16_t* __restrict__ img_all,
    uint16_t* __restrict__ mean,
    int n_img,
    int npixels,
    float k,
    int it_n
) {
    float var;
    for (int it = 0; it < it_n; it++) {
        #pragma omp parallel for
        for (uint64_t idx = 0; idx < npixels; idx++) {
            mean_imgs(img_all, mean, idx, n_img, npixels);
            var_imgs(var, mean, img_all, idx, n_img, npixels);
            filter_pixels(mean, var, img_all, idx, k, n_img, npixels);
        }
    }
    #pragma omp parallel for
    for (uint64_t idx = 0; idx < npixels; idx++)
        mean_imgs(img_all, mean, idx, n_img, npixels);
}



// Find the highest in-bounds value (max value <= upper_bound)
// Falls back to upper_bound if no in-bounds value found (avoids erasing bright features)
uint16_t inline find_highest_inbound(uint16_t* __restrict__ img_all, uint64_t i, uint64_t npixels, int n_img, uint16_t upper_bound) {
    uint16_t max = 0;
    for (int j = 0; j < n_img; j++) {
        auto current = img_all[npixels * j + i];
        if (current <= upper_bound && current > max)
            max = current;
    }
    return max > 0 ? max : upper_bound;
}

// Find the lowest in-bounds value (min value >= lower_bound)
uint16_t inline find_lowest_inbound(uint16_t* __restrict__ img_all, uint64_t i, uint64_t npixels, int n_img, uint16_t lower_bound) {
    uint16_t min = 65535;
    for (int j = 0; j < n_img; j++) {
        auto current = img_all[npixels * j + i];
        if (current >= lower_bound && current < min)
            min = current;
    }
    return min < 65535 ? min : 0;
}


void inline replace_oob_pixels(uint16_t* __restrict__ mean, float std, uint16_t* __restrict__ img_all, uint64_t idx, float k_low, float k_high, int n_img, int npixels) {
    float lb = (float)mean[idx] - k_low * std;
    float ub = (float)mean[idx] + k_high * std;
    uint16_t lower_bound = lb > 0 ? lb : 0;
    uint16_t upper_bound = ub < 65535 ? ub : 65535;
    
    for (int j = 0; j < n_img; j++) {
        uint64_t base = npixels*j;
        uint16_t val = img_all[base + idx];
        if (val > 0) {
            // Winsorizing: replace outlier with nearest in-bounds value from other images
            if (val > upper_bound)
                img_all[base + idx] = find_highest_inbound(img_all, idx, npixels, n_img, upper_bound);
            else if (val < lower_bound)
                img_all[base + idx] = find_lowest_inbound(img_all, idx, npixels, n_img, lower_bound);
        }
    }
}


void inline clip_oob_pixels(uint16_t* __restrict__ mean, float std, uint16_t* __restrict__ img_all, uint64_t idx, float k_low, float k_high, int n_img, int npixels) {
    float lb = (float)mean[idx] - k_low * std;
    float ub = (float)mean[idx] + k_high * std;
    uint16_t lower_bound = lb > 0 ? lb : 0;
    uint16_t upper_bound = ub < 65535 ? ub : 65535;
    
    for (int j = 0; j < n_img; j++) {
        uint64_t base = npixels*j;
        uint16_t val = img_all[base + idx];
        if (val > 0) {
            // Winsorizing: replace outlier with nearest in-bounds value from other images
            if (val > upper_bound)
                img_all[base + idx] = upper_bound;
            else if (val < lower_bound)
                img_all[base + idx] = lower_bound;
        }
    }
}

void inline set_to_mean_oob_pixels(uint16_t* __restrict__ mean, float std, uint16_t* __restrict__ img_all, uint64_t idx, float k_low, float k_high, int n_img, int npixels) {
    float lb = (float)mean[idx] - k_low * std;
    float ub = (float)mean[idx] + k_high * std;
    uint16_t lower_bound = lb > 0 ? lb : 0;
    uint16_t upper_bound = ub < 65535 ? ub : 65535;
    
    for (int j = 0; j < n_img; j++) {
        uint64_t base = npixels*j;
        uint16_t val = img_all[base + idx];
        if (val > 0 && (val > upper_bound || val < lower_bound))
            img_all[base + idx] = mean[idx];
    }
}

void winsorized_sigma_clipping_cpu(
    uint16_t* __restrict__ img_all,
    uint16_t* __restrict__ mean,
    int n_img,
    int npixels,
    float k_low,
    float k_high,
    float k1_low,
    float k2_high,
    float conv_tolerance
) {
    float* __restrict__ sig = (float*)malloc(npixels * sizeof(float));
    uint16_t* __restrict__ prev_mean = (uint16_t*)malloc(npixels * sizeof(uint16_t));
    bool* __restrict__ converged = (bool*)malloc(npixels * sizeof(bool));
    int max_iterations = 100;

    // Phase 1: median + MAD (robust to outliers)
    #pragma omp parallel for
    for (uint64_t idx = 0; idx < npixels; idx++) {
        median_imgs(img_all, mean, idx, n_img, npixels);
        mad_sigma(sig[idx], img_all, mean, idx, n_img, npixels);
        set_to_mean_oob_pixels(mean, sig[idx], img_all, idx, k_low, k_high, n_img, npixels);
    }

    // Phase 2: iterate with mean + classical std — per-pixel convergence (matching GPU)
    // Each pixel converges independently: when |new_mean - old_mean| < tolerance, it stops iterating
    memset(converged, 0, npixels * sizeof(bool));

    for (int iter = 0; iter < max_iterations; iter++) {
        #pragma omp parallel for
        for (uint64_t idx = 0; idx < npixels; idx++) {
            if (converged[idx]) continue;

            prev_mean[idx] = mean[idx];
            mean_imgs(img_all, mean, idx, n_img, npixels);
            var_imgs(sig[idx], mean, img_all, idx, n_img, npixels);
            sig[idx] = sqrtf(sig[idx]);
            clip_oob_pixels(mean, sig[idx], img_all, idx, k1_low, k2_high, n_img, npixels);

            if (fabsf((float)mean[idx] - (float)prev_mean[idx]) < conv_tolerance)
                converged[idx] = true;
        }

        // Early exit if all pixels have converged
        bool all_converged = true;
        for (int i = 0; i < npixels; i++) {
            if (!converged[i]) {
                all_converged = false;
                break;
            }
        }
        if (all_converged) break;
    }

    // Final mean computation
    #pragma omp parallel for
    for (uint64_t idx = 0; idx < npixels; idx++)
        mean_imgs(img_all, mean, idx, n_img, npixels);

    free(sig);
    free(prev_mean);
    free(converged);
}

void simple_winsorized_sigma_clipping_cpu(
    uint16_t* __restrict__ img_all,
    uint16_t* __restrict__ mean,
    int numImages,
    int npixels,
    float k_low,
    float k_high,
    int iterations
) {
    float std, var;
    for (int i = 0; i < iterations; i++) {
        #pragma omp parallel for
        for (uint64_t idx = 0; idx < npixels; idx++) {
            mean_imgs(img_all, mean, idx, numImages, npixels);
            var_imgs(var, mean, img_all, idx, numImages, npixels);
            std = sqrtf(var);
            clip_oob_pixels(mean, std, img_all, idx, k_low, k_high, numImages, npixels);
        }
    }
    #pragma omp parallel for
    for (uint64_t idx = 0; idx < npixels; idx++)
        mean_imgs(img_all, mean, idx, numImages, npixels);
}
