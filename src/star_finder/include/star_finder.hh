#ifndef STAR_FINDER_HH
#define STAR_FINDER_HH

#include <stdint.h>

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"

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
    TR_FAST_ADAPTIVE,
    OTSU_CENTRALIZED
};

struct threshold_params {
    threshold_type type = OTSU_CENTRALIZED;
    uint16_t threshold;
    uint16_t window_size = 101;
    uint16_t reduce_factor;
    float threshold_scale = 0.7f;
};

// GPU

void to_grayscale_planar_gpu(
    const uint16_t* __restrict__ img,
    uint16_t* __restrict__ img_gray,
    uint64_t npixels
);

void compute_threshold_gpu(
    const uint16_t* __restrict__ img,
    uint8_t* __restrict__ out_img,
    uint64_t width,
    uint64_t height,
    threshold_params params
);

void detect_stars_gpu(
    const uint8_t* __restrict__ threshold_image,
    uint64_t width,
    uint64_t height,
    uint16_t max_star_size,
    uint16_t min_star_size,
    star *d_stars,
    uint32_t *d_num_stars,
    uint32_t max_stars
);

void populate_star_details_gpu(
    star_detail* stars_details,
    star* stars,
    uint32_t n_stars,
    const uint16_t* img_rgb,
    const uint16_t* img_gray,
    uint64_t width,
    uint64_t npixels
);


// CPU
void to_grayscale_planar_cpu(
    const uint16_t* image,
    uint16_t* gray_image,
    uint64_t npixels
);

void compute_threshold_cpu(
    const uint16_t* img,
    uint8_t* out_img,
    uint64_t width,
    uint64_t height,
    threshold_params params
);

void detect_stars_cpu(
    const uint8_t* threshold_image,
    uint64_t width,
    uint64_t height,
    uint16_t max_star_size,
    uint16_t min_star_size,
    star *stars,
    uint32_t &num_stars,
    uint32_t max_stars
);

void populate_star_details(
    star_detail *stars_details,
    star *stars,
    uint32_t n_stars,
    const uint16_t* img_rgb,
    const uint16_t* img_gray,
    uint64_t width,
    uint64_t npixels
);

void draw_stars(
    uint16_t* img,
    uint64_t  width,
    const star *stars,
    uint32_t  n_stars
);


// CPU only

bool build_star_descriptors(
    const star_detail *stars,
    uint32_t count,
    long width,
    long height,
    std::vector<cv::KeyPoint> &keypoints,
    cv::Mat &descriptors
);

bool build_star_descriptors_generalized(
    const star_detail *stars,
    uint32_t count,
    long width,
    long height,
    int neighbors,
    std::vector<cv::KeyPoint> &keypoints,
    cv::Mat &descriptors
);


// Warping

/* estimate the trasformation between two images
 * first it find the good matches between the two keypoints vectors
 * then uses the function cv::estimateAffinePartial2D
*/
cv::Mat estimate_affine_partial_stars(
    const std::vector<cv::KeyPoint> &keypoints1,
    const cv::Mat                   &descriptors1, 
    const std::vector<cv::KeyPoint> &keypoints2,
    const cv::Mat                   &descriptors2,
    float                           ratio_threshold = 0.7F,
    std::vector<cv::DMatch>*        inlier_matches = nullptr
);


void warp_affine_planar_cpu(
    const uint16_t  *source,
    uint16_t        *dest,
    const cv::Mat   &affine_2x3,
    int64_t         width, 
    int64_t         height
);

void warp_affine_planar_gpu(
    const uint16_t  *source,
    uint16_t        *dest,
    const cv::Mat   &affine_2x3,
    int64_t        width,
    int64_t        height
);


#endif