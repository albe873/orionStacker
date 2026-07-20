#ifndef STACKER_H
#define STACKER_H

void alfa_sigma(
    u_int16_t *img_all,
    u_int16_t *mean,
    u_int16_t numImages,
    u_int64_t npixels,
    float k,
    u_int16_t s
);

void alfa_sigma_cpu(
    u_int16_t *img_all,
    u_int16_t *mean,
    float *std,
    int numImages,
    int npixels,
    float kappa,
    int sigma
);

#endif // STACKER_H