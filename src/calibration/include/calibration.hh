#ifndef CUDA_DEVICE_CALIBRATION_HH
#define CUDA_DEVICE_CALIBRATION_HH

#include <cstdint>

// =========================
// GPU CALIBRATION FUNCTIONS

void masterBias_gpu(
    const uint16_t* bias_all,
    float*  master_bias,
    uint64_t width,
    uint64_t height,
    int bias_count
);

void masterDark_gpu(
    const uint16_t* dark_all,
    const float* master_bias,
    float* master_dark,
    uint64_t width,
    uint64_t height,
    int dark_count
);

void masterFlat_gpu(
    const uint16_t* flat_all,
    const float* master_bias,
    float* master_flat,
    uint64_t width,
    uint64_t height,
    int flat_count
);

void calibrateLights_gpu(
    const uint16_t* light_all,
    const float* master_bias,
    const float* master_dark,
    const float* master_flat,
    uint16_t* calib_all,
    uint64_t width,
    uint64_t height,
    int light_count
);



// =========================
// CPU CALIBRATION FUNCTIONS

void masterBias_cpu(
    const uint16_t* bias_all,
    float* master_bias,
    uint64_t width,
    uint64_t height,
    int bias_count
);

void masterDark_cpu(
    const uint16_t* dark_all,
    const float* master_bias,
    float* master_dark,
    uint64_t width,
    uint64_t height,
    int dark_count
);

void masterFlat_cpu(
    const uint16_t* flat_all,
    const float* master_bias,
    float* master_flat,
    uint64_t width,
    uint64_t height,
    int flat_count
);

void calibrateLights_cpu(
    const uint16_t* light_all,
    const float* master_bias,
    const float* master_dark,
    const float* master_flat,
    uint16_t* calib_all,
    uint64_t width,
    uint64_t height,
    int light_count
);

#endif