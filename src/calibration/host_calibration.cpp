#include <stdint.h>
#include <algorithm>
#include <cmath>
#include "calibration.h"

// CPU clamp functions
static inline uint16_t clamp_u16_from_f32(float x) {
    if (x <= 0.0f) return 0;
    if (x >= 65535.0f) return 65535;
    return static_cast<uint16_t>(x + 0.5f);
}

static inline uint16_t clamp_u16_from_u64(uint64_t x) {
    return (x > 65535ULL) ? 65535 : static_cast<uint16_t>(x);
}


void masterBias_cpu(const uint16_t* __restrict__ bias_all, uint16_t* __restrict__ master_bias, long width, long height, int bias_count) {
    uint64_t npixels = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
    
    for (uint64_t idx = 0; idx < npixels; idx++) {
        int c = 0;
        float acc = 0.0f;
        for (int i = 0; i < bias_count; i++) {
            float val = static_cast<float>(bias_all[i * npixels + idx]);
            if (val > 0) {
                c++;
                acc += val;
            }
        }
        master_bias[idx] = (c > 0) ? clamp_u16_from_f32(acc / static_cast<float>(c)) : 0;
    }
}

void masterDark_cpu(const uint16_t* __restrict__ dark_all, const uint16_t* __restrict__ master_bias, uint16_t* __restrict__ master_dark, long width, long height, int dark_count) {
    uint64_t npixels = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
    
    // For each pixel, subtract master bias and compute mean excluding values <= 0
    for (uint64_t idx = 0; idx < npixels; idx++) {
        int c = 0;
        float acc = 0.0f;
        for (int i = 0; i < dark_count; i++) {
            float val = static_cast<float>(dark_all[i * npixels + idx]) - static_cast<float>(master_bias[idx]);
            if (val > 0) {
                c++;
                acc += val;
            }
        }
        master_dark[idx] = (c > 0) ? clamp_u16_from_f32(acc / static_cast<float>(c)) : 0;
    }
}

void masterFlat_cpu(const uint16_t* __restrict__ flat_all, const uint16_t* __restrict__ master_bias, uint16_t* __restrict__ master_flat, long width, long height, int flat_count) {
    uint64_t npixels = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
    
    // For each pixel, subtract master bias and compute mean excluding values <= 0
    for (uint64_t idx = 0; idx < npixels; idx++) {
        int c = 0;
        float acc = 0.0f;
        for (int i = 0; i < flat_count; i++) {
            float val = static_cast<float>(flat_all[i * npixels + idx]) - static_cast<float>(master_bias[idx]);
            if (val > 0) {
                c++;
                acc += val;
            }
        }
        master_flat[idx] = (c > 0) ? clamp_u16_from_f32(acc / static_cast<float>(c)) : 0;
    }
    
    // Normalize master flat (divide each pixel by the mean value of master flat)
    // Calculate mean value of master flat
    uint64_t sum = 0;
    for (uint64_t i = 0; i < npixels; i++) {
        sum += master_flat[i];
    }
    float mean_val = (sum > 0) ? (static_cast<float>(sum) / static_cast<float>(npixels)) : 0.0f;
    
    // Normalize master flat
    if (mean_val > 0.0f) {
        const float norm_scale = 65535.0f / mean_val;
        for (uint64_t i = 0; i < npixels; i++) {
            // Store normalized flat in fixed-point scale [0, 65535].
            float scaled = static_cast<float>(master_flat[i]) * norm_scale;
            master_flat[i] = clamp_u16_from_f32(scaled);
        }
    } else {
        // If mean_val == 0, leave master_flat as is (all 0), to avoid division by zero
        // In calibrateLights_cpu, flat == 0 will be handled by setting calib to 0
    }
}

void calibrateLights_cpu(const uint16_t* __restrict__ light_all, const uint16_t* __restrict__ master_bias, const uint16_t* __restrict__ master_dark, const uint16_t* __restrict__ master_flat, uint16_t* __restrict__ calib_all, long width, long height, int light_count) {
    uint64_t npixels = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
    
    // For each pixel of each light image: calibrate = (light - master_bias - master_dark) / master_flat
    for (int i = 0; i < light_count; i++) {
        for (uint64_t idx = 0; idx < npixels; idx++) {
            float val = static_cast<float>(light_all[i * npixels + idx]) 
                      - static_cast<float>(master_bias[idx]) 
                      - static_cast<float>(master_dark[idx]);
            if (val < 0.0f)
                val = 0.0f;
            
            uint16_t flat = master_flat[idx];
            if (flat > 0) {
                // master_flat is normalized in [0, 65535], so multiply by 65535 before division.
                float scaled = (val * 65535.0f) / static_cast<float>(flat);
                calib_all[i * npixels + idx] = clamp_u16_from_f32(scaled);
            } else {
                calib_all[i * npixels + idx] = 0;
            }
        }
    }
}