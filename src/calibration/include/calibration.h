#ifndef CUDA_DEVICE_DEBAYER_H
#define CUDA_DEVICE_DEBAYER_H

#include <stdint.h>
#include <sys/types.h>

void masterBias(u_int16_t *bias_all, u_int16_t *master_bias, long width, long height, int bias_count);

#endif