#ifndef STACKER_H
#define STACKER_H


void compute_alfa_sigma(
    u_int16_t **image,
    u_int16_t *mean,
    u_int16_t numImages,
    u_int64_t npixels,
    float k,
    u_int16_t s
);

void compute_alfa_sigma_cpu (
    u_int16_t **image,
    u_int16_t *mean,
    u_int16_t numImages,
    u_int64_t npixels,
    float k,
    u_int16_t s
);


#endif // STACKER_H