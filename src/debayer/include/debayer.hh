#ifndef CUDA_DEVICE_DEBAYER_HH
#define CUDA_DEVICE_DEBAYER_HH

#include <cstdint>

// ========================
// GPU

void demosaic_bilinear_rggb_gpu(
    const uint16_t* gray_all,
    uint16_t* rgb_all,
    long width, long height,
    uint16_t image_count
);

void demosaic_mhc_rggb_gpu(
    const uint16_t* gray_all,
    uint16_t * rgb_all,
    long width,
    long height,
    uint16_t image_count
);


// =========================
// CPU
void demosaic_bilinear_rggb_cpu(
    const uint16_t* gray_all,
    uint16_t* rgb_all,
    long width, long height,
    uint16_t image_count
);

void demosaic_mhc_rggb_cpu(
    const uint16_t* gray_all,
    uint16_t * rgb_all,
    long width,
    long height,
    uint16_t image_count
);

#endif