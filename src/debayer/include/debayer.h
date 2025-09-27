#ifndef CUDA_DEVICE_DEBAYER_H
#define CUDA_DEVICE_DEBAYER_H

#include <stdint.h>
#include "MHC_filters.h"
#include "MHC_apply.h"
#include <sys/types.h>

void demosaic_bilinear_rggb(
    const u_int16_t *gray_all,
    u_int16_t *rgb_all,
    long width, long height,
    u_int16_t image_count
);

void demosaic_mhc_rggb(
    const u_int16_t * __restrict__ gray_all,
    u_int16_t * __restrict__ rgb_all,
    long width,
    long height,
    u_int16_t image_count
);

#endif