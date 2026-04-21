#ifndef STAR_FINDER_H
#define STAR_FINDER_H

#include <stdint.h>
#include <sys/types.h>

struct star {
    uint64_t start_x;
    uint64_t start_y;
    uint32_t size_x;
    uint32_t size_y;
};

struct star_detail {
    double x;
    double y;
    double b_red;
    double b_green;
    double b_blue;
    double b;
};

inline void init_star_detail(star_detail *star) {
    star->x = 0;
    star->y = 0;
    star->b_red = 0;
    star->b_green = 0;
    star->b_blue = 0;
    star->b = 0;
}

enum threshold_type {
    TR_SIMPLE,
    TR_ADAPTIVE,
    TR_FAST_ADAPTIVE
};

struct threshold_params {
    threshold_type type;
    u_int16_t threshold;
    u_int16_t window_size;
    u_int16_t reduce_factor;
};

// GPU

void to_grayscale_planar_gpu(const u_int16_t *img, u_int16_t *img_gray, u_int64_t npixels);

void compute_threshold_gpu(const u_int16_t *img, u_int8_t *out_img,
                           u_int64_t width, u_int64_t height,
                           threshold_params params);

void detect_stars_gpu(const u_int8_t *threshold_image, u_int64_t width, u_int64_t height,
                      u_int16_t max_star_size, u_int16_t min_star_size,
                      star *d_stars, u_int32_t *d_num_stars, u_int32_t max_stars);

void populate_star_details_gpu(star_detail *stars_details, star *stars, u_int32_t n_stars,
                               const u_int16_t *img_rgb, const u_int16_t *img_gray,
                               u_int64_t width, u_int64_t npixels);


// CPU
void to_grayscale_planar(const uint16_t *image, uint16_t *gray_image, uint64_t npixels);

void compute_threshold(const u_int16_t *img, u_int8_t *out_img,
                       u_int64_t width, u_int64_t height,
                       threshold_params params);

void detect_stars(const uint8_t *threshold_image, uint64_t width, uint64_t height,
                  uint16_t max_star_size, uint16_t min_star_size,
                  star *stars, uint32_t &num_stars, uint32_t max_stars);

void populate_star_details(star_detail *stars_details, star *stars, u_int32_t n_stars,
                               const u_int16_t *img_rgb, const u_int16_t *img_gray,
                               u_int64_t width, u_int64_t npixels);

void draw_stars(u_int16_t *img, u_int64_t width, star *stars, u_int32_t n_stars);

#endif // STAR_FINDER_H