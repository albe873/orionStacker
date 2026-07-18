#ifndef CUDA_DEVICE_CALIBRATION_H
#define CUDA_DEVICE_CALIBRATION_H

#include <stdint.h>
#include <sys/types.h>

// =========================
// GPU CALIBRATION FUNCTIONS

void masterBias(
    const uint16_t* bias_all,
    uint16_t*  master_bias,
    long width,
    long height,
    int bias_count
);

void masterDark(
    const u_int16_t* dark_all,
    const u_int16_t* master_bias,
    u_int16_t* master_dark,
    long width,
    long height,
    int dark_count
);

void masterFlat(
    const u_int16_t* flat_all,
    const u_int16_t* master_bias,
    u_int16_t* master_flat,
    long width,
    long height,
    int flat_count
);

void calibrateLights(
    const u_int16_t* light_all,
    const u_int16_t* master_bias,
    const u_int16_t* master_dark,
    const u_int16_t* master_flat,
    u_int16_t* calib_all,
    long width,
    long height,
    int light_count
);



// =========================
// CPU CALIBRATION FUNCTIONS

void masterBias_cpu(
    const u_int16_t* bias_all,
    u_int16_t* master_bias,
    long width,
    long height,
    int bias_count
);

void masterDark_cpu(
    const u_int16_t* dark_all,
    const u_int16_t* master_bias,
    u_int16_t* master_dark,
    long width,
    long height,
    int dark_count
);

void masterFlat_cpu(
    const u_int16_t* flat_all,
    const u_int16_t* master_bias,
    u_int16_t* master_flat,
    long width,
    long height,
    int flat_count
);

void calibrateLights_cpu(
    const u_int16_t* light_all,
    const u_int16_t* master_bias,
    const u_int16_t* master_dark,
    const u_int16_t* master_flat,
    u_int16_t* calib_all,
    long width,
    long height,
    int light_count
);

#endif