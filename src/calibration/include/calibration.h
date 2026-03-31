#ifndef CUDA_DEVICE_DEBAYER_H
#define CUDA_DEVICE_DEBAYER_H

#include <stdint.h>
#include <sys/types.h>

void masterBias(float *bias_all, float *master_bias, long width, long height, int bias_count);

void masterDark(float *dark_all, float *master_bias, float *master_dark, long width, long height, int dark_count);

void masterFlat(float *flat_all, float *master_bias, float *master_flat, long width, long height, int flat_count);

void calibrateLights(float *light_all, float *master_bias, float *master_dark, float *master_flat, float *calib_all, long width, long height, int light_count);

#endif