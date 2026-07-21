#include "star_finder.h"

#include "opencv2/imgproc.hpp"

void warp_affine_planar_cpu(const u_int16_t *source, u_int16_t *dest,
                            const cv::Mat &affine_2x3, long width, long height) {
    long npixels = width * height;
    cv::Mat src_r(height, width, CV_16UC1, const_cast<u_int16_t*>(source));
    cv::Mat src_g(height, width, CV_16UC1, const_cast<u_int16_t*>(source + npixels));
    cv::Mat src_b(height, width, CV_16UC1, const_cast<u_int16_t*>(source + 2*npixels));

    cv::Mat dst_r(height, width, CV_16UC1, dest);
    cv::Mat dst_g(height, width, CV_16UC1, dest + npixels);
    cv::Mat dst_b(height, width, CV_16UC1, dest + 2*npixels);

    cv::warpAffine(src_r, dst_r, affine_2x3, cv::Size(width, height),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::warpAffine(src_g, dst_g, affine_2x3, cv::Size(width, height),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::warpAffine(src_b, dst_b, affine_2x3, cv::Size(width, height),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
}