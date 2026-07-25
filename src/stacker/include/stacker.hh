#ifndef STACKER_HH
#define STACKER_HH

#include <cstdint>


// =============================================================
// GPU

void alfa_sigma_gpu(
    uint16_t *img_all,
    uint16_t *mean,
    int numImages,
    uint64_t npixels,
    float kappa = 3,
    int iterations = 5
);

void simple_winsorized_sigma_clipping_gpu(
    uint16_t *img_all,
    uint16_t *mean,
    int numImages,
    int npixels,
    float kappa_low  = 4.0f,
    float kappa_high = 1.5f,
    int iterations   = 5
);

void winsorized_sigma_clipping_gpu(
    uint16_t* __restrict__ img_all,
    uint16_t* __restrict__ mean,
    int n_img,
    int npixels,
    float k_low   = 4.0f, // first phase
    float k_high  = 2.0f, // first phase
    float k1_low  = 3.5f, // second phase
    float k2_high = 3.0f, // second phase
    float conv_tolerance = 1.0F
);

// ==============================================================
// CPU


void alfa_sigma_cpu(
    uint16_t *img_all,
    uint16_t *mean,
    int numImages,
    int npixels,
    float kappa = 3,
    int iterations = 5
);



void simple_winsorized_sigma_clipping_cpu(
    uint16_t *img_all,
    uint16_t *mean,
    int numImages,
    int npixels,
    float kappa_low  = 4.0f,
    float kappa_high = 1.5f,
    int iterations   = 5
);

void winsorized_sigma_clipping_cpu(
    uint16_t* __restrict__ img_all,
    uint16_t* __restrict__ mean,
    int n_img,
    int npixels,
    float k_low   = 4.0f, // first phase
    float k_high  = 2.0f, // first phase
    float k1_low  = 3.5f, // second phase
    float k2_high = 3.0f, // second phase
    float conv_tolerance = 1.0F
);

#endif