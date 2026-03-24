#ifndef CUDA_DEVICE_DEBAYER_H
#define CUDA_DEVICE_DEBAYER_H

#include <stdint.h>
#include <sys/types.h>

void masterBias(u_int16_t *bias_all, u_int16_t *master_bias, long width, long height, int bias_count);

void masterDark(u_int16_t *dark_all, u_int16_t *master_bias, u_int16_t *master_dark, long width, long height, int dark_count);

void masterFlat(u_int16_t *flat_all, u_int16_t *master_bias, u_int16_t *master_flat, long width, long height, int flat_count);

void calibrateLights(u_int16_t *light_all, u_int16_t *master_bias, u_int16_t *master_dark, u_int16_t *master_flat, u_int16_t *calib_all, long width, long height, int light_count);

#endif