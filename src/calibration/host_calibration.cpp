#include "calibration.hh"

#include <algorithm>
#include <cmath>

// CPU clamp functions
inline uint16_t clamp_u16_from_f32(float x) {
    if (x <= 0.0f) return 0;
    if (x >= 65535.0f) return 65535;
    return static_cast<uint16_t>(x + 0.5f);
}

// ===== Helper functions =====

inline void mean_w_sub(const uint16_t* __restrict__ all, const float* __restrict__ sub, float* __restrict__ res, int img_n, uint64_t npixels) {
    #pragma parallel for
    for (uint64_t idx=0; idx < npixels; idx++) {
        double acc = 0.0D;
        for (int i = 0; i < img_n; i++)
            acc = acc + all[npixels*i + idx] - sub[idx];
        res[idx] = acc / img_n;
    }
}
inline void mean(const uint16_t* __restrict__ all, float* __restrict__ res, int img_n, uint64_t npixels) {
    #pragma parallel for
    for (uint64_t idx=0; idx < npixels; idx++) {
        double acc = 0.0D;
        for (int i = 0; i < img_n; i++)
            acc = acc + all[npixels*i + idx];
        res[idx] = acc / img_n;
    }
}


// ===== Masters computing =====

void masterBias_cpu(const uint16_t* __restrict__ bias_all, float* __restrict__ master_bias, uint64_t width, uint64_t height, int bias_count) {
    uint64_t npixels = width * height;
    if (bias_count < 0)
        return;
    mean(bias_all, master_bias, bias_count, npixels);
}

void masterDark_cpu(const uint16_t* __restrict__ dark_all, const float* __restrict__ master_bias, float* __restrict__ master_dark, uint64_t width, uint64_t height, int dark_count) {
    uint64_t npixels = width * height;
    if (dark_count < 0)
        return;
    //mean_w_sub(dark_all, master_bias, master_dark, dark_count, npixels);
    mean(dark_all, master_dark, dark_count, npixels);
}

void masterFlat_cpu(const uint16_t* __restrict__ flat_all, const float* __restrict__ master_bias, float* __restrict__ master_flat, uint64_t width, uint64_t height, int flat_count) {
    uint64_t npixels = width *height;
    if (flat_count < 0)
        return;
    //mean_w_sub(flat_all, master_bias, master_flat, flat_count, npixels);
    mean(flat_all, master_flat, flat_count, npixels);
}


// ===== calibration =====

void calibrateLights_cpu(const uint16_t* __restrict__ light_all, const float* __restrict__ master_bias, const float* __restrict__ master_dark, const float* __restrict__ master_flat, uint16_t* __restrict__ calib_all, uint64_t width, uint64_t height, int light_count) {
    uint64_t npixels = width * height;

    float sum = 0.0F;
    
    #pragma omp parallel for reduction(+:sum)
    for (int64_t i = 0; i < npixels; i++)
        sum += master_flat[i];
    float master_flat_mean_val = sum / (float)npixels;
    
    for (int i = 0; i < light_count; i++) {
        #pragma omp parallel for
        for (uint64_t idx = 0; idx < npixels; idx++) {
 
            float light_val = (float)light_all[i * npixels + idx];
            float master_bias_val = master_bias[idx];
            
            //float numerator = light_val - master_dark[idx] - master_bias[idx];
            float numerator = light_val - master_dark[idx];
            //float denominator = master_flat[idx] / master_flat_mean_val;
            float denominator = (master_flat[idx] - master_bias[idx]) / master_flat_mean_val;

            calib_all[i * npixels + idx] = clamp_u16_from_f32(numerator / denominator);
        }
    }
}