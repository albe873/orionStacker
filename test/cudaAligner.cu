#include "fits_helper.h"
#include "cuda_helper.h"
#include "common.h"

#include "star_finder.h"
#include "debayer.h"

#include <stdio.h>
#include <getopt.h>
#include <unistd.h>
#include <cstring>
#include <vector>
#include <array>
#include <algorithm>
#include <limits>
#include <cmath>
#include <string>
#include <cstdint>
#include <filesystem>
#include <utility>

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"

namespace {

cv::Mat normalize_u16_to_u8(const u_int16_t *img, int width, int height) {
    cv::Mat src_u16(height, width, CV_16UC1, const_cast<u_int16_t*>(img));
    cv::Mat out_u8;
    cv::normalize(src_u16, out_u8, 0, 255, cv::NORM_MINMAX, CV_8U);
    return out_u8;
}

u_int16_t min_u16_value(const u_int16_t *img, u_int64_t npixels) {
    if (npixels == 0) {
        return 0;
    }

    u_int16_t min_v = std::numeric_limits<u_int16_t>::max();
    for (u_int64_t i = 0; i < npixels; i++) {
        min_v = std::min(min_v, img[i]);
    }
    return min_v;
}

inline float clamp(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

void print_u16_stats(const char *label, const u_int16_t *img, u_int64_t npixels) {
    if (npixels == 0) {
        printf("%s: empty\n", label);
        return;
    }

    u_int16_t min_v = std::numeric_limits<u_int16_t>::max();
    u_int16_t max_v = 0;
    uint64_t sum = 0;
    for (u_int64_t i = 0; i < npixels; i++) {
        const u_int16_t v = img[i];
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
        sum += static_cast<uint64_t>(v);
    }
    const double mean = static_cast<double>(sum) / static_cast<double>(npixels);
    printf("%s: min=%u max=%u mean=%.2f\n", label, static_cast<unsigned>(min_v), static_cast<unsigned>(max_v), mean);
}

void print_u8_threshold_stats(const char *label, const u_int8_t *img, u_int64_t npixels) {
    if (npixels == 0) {
        printf("%s: empty\n", label);
        return;
    }

    u_int64_t zero = 0;
    u_int64_t non_zero = 0;
    u_int8_t min_v = std::numeric_limits<u_int8_t>::max();
    u_int8_t max_v = 0;
    for (u_int64_t i = 0; i < npixels; i++) {
        const u_int8_t v = img[i];
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
        if (v == 0) {
            zero++;
        } else {
            non_zero++;
        }
    }

    const double zero_pct = 100.0 * static_cast<double>(zero) / static_cast<double>(npixels);
    const double non_zero_pct = 100.0 * static_cast<double>(non_zero) / static_cast<double>(npixels);
    printf("%s: min=%u max=%u zero=%llu (%.2f%%) non_zero=%llu (%.2f%%)\n",
           label,
           static_cast<unsigned>(min_v),
           static_cast<unsigned>(max_v),
           static_cast<unsigned long long>(zero),
           zero_pct,
           static_cast<unsigned long long>(non_zero),
           non_zero_pct);
}

bool build_star_descriptors(const star_detail *stars,
                            u_int32_t count,
                            long width,
                            long height,
                            std::vector<cv::KeyPoint> &keypoints,
                            cv::Mat &descriptors) {
    keypoints.clear();
    descriptors.release();

    if (count < 3) {
        return false;
    }

    const float image_diag = std::sqrt(static_cast<float>(width * width + height * height));
    if (image_diag <= 0.0f) {
        return false;
    }

    std::vector<std::array<float, 3>> rows;
    rows.reserve(count);

    for (u_int32_t i = 0; i < count; i++) {
        const float xi = static_cast<float>(stars[i].x);
        const float yi = static_cast<float>(stars[i].y);

        float best_d2_1 = std::numeric_limits<float>::max();
        float best_d2_2 = std::numeric_limits<float>::max();
        int idx1 = -1;
        int idx2 = -1;

        for (u_int32_t j = 0; j < count; j++) {
            if (j == i) {
                continue;
            }

            const float dx = static_cast<float>(stars[j].x) - xi;
            const float dy = static_cast<float>(stars[j].y) - yi;
            const float d2 = dx * dx + dy * dy;

            if (d2 < best_d2_1) {
                best_d2_2 = best_d2_1;
                idx2 = idx1;
                best_d2_1 = d2;
                idx1 = static_cast<int>(j);
            } else if (d2 < best_d2_2) {
                best_d2_2 = d2;
                idx2 = static_cast<int>(j);
            }
        }

        if (idx1 < 0 || idx2 < 0) {
            continue;
        }

        const float x1 = static_cast<float>(stars[idx1].x);
        const float y1 = static_cast<float>(stars[idx1].y);
        const float x2 = static_cast<float>(stars[idx2].x);
        const float y2 = static_cast<float>(stars[idx2].y);

        const float v1x = x1 - xi;
        const float v1y = y1 - yi;
        const float v2x = x2 - xi;
        const float v2y = y2 - yi;

        const float d1 = std::sqrt(v1x * v1x + v1y * v1y);
        const float d2 = std::sqrt(v2x * v2x + v2y * v2y);
        if (d1 <= 1e-6f || d2 <= 1e-6f) {
            continue;
        }

        float cos_angle = (v1x * v2x + v1y * v2y) / (d1 * d2);
        cos_angle = clamp(cos_angle, -1.0f, 1.0f);
        const float angle_norm = std::acos(cos_angle) / static_cast<float>(CV_PI);

        rows.push_back({d1 / image_diag, d2 / image_diag, angle_norm});
        keypoints.emplace_back(cv::Point2f(xi, yi), 5.0f);
    }

    if (rows.size() < 4) {
        keypoints.clear();
        return false;
    }

    descriptors = cv::Mat(static_cast<int>(rows.size()), 3, CV_32F);
    for (int r = 0; r < descriptors.rows; r++) {
        descriptors.at<float>(r, 0) = rows[r][0];
        descriptors.at<float>(r, 1) = rows[r][1];
        descriptors.at<float>(r, 2) = rows[r][2];
    }

    return true;
}

bool build_star_descriptors_generalized(const star_detail *stars,
                                        u_int32_t count,
                                        long width,
                                        long height,
                                        int neighbors,
                                        std::vector<cv::KeyPoint> &keypoints,
                                        cv::Mat &descriptors) {
    keypoints.clear();
    descriptors.release();

    if (neighbors < 2) {
        return false;
    }

    if (count < static_cast<u_int32_t>(neighbors + 1)) {
        return false;
    }

    const float image_diag = std::sqrt(static_cast<float>(width * width + height * height));
    if (image_diag <= 0.0f) {
        return false;
    }

    const int descriptor_dim = 2 * neighbors - 1;
    std::vector<float> all_rows;
    all_rows.reserve(static_cast<size_t>(count) * static_cast<size_t>(descriptor_dim));

    for (u_int32_t i = 0; i < count; i++) {
        const float xi = static_cast<float>(stars[i].x);
        const float yi = static_cast<float>(stars[i].y);

        std::vector<std::pair<float, int>> d2_with_idx;
        d2_with_idx.reserve(count > 0 ? count - 1 : 0);

        for (u_int32_t j = 0; j < count; j++) {
            if (j == i) {
                continue;
            }

            const float dx = static_cast<float>(stars[j].x) - xi;
            const float dy = static_cast<float>(stars[j].y) - yi;
            const float d2 = dx * dx + dy * dy;
            d2_with_idx.emplace_back(d2, static_cast<int>(j));
        }

        if (static_cast<int>(d2_with_idx.size()) < neighbors) {
            continue;
        }

        std::partial_sort(
            d2_with_idx.begin(),
            d2_with_idx.begin() + neighbors,
            d2_with_idx.end(),
            [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
                return a.first < b.first;
            }
        );

        std::vector<cv::Point2f> vectors;
        vectors.reserve(neighbors);

        bool valid = true;
        for (int k = 0; k < neighbors; k++) {
            const int idx = d2_with_idx[k].second;
            const float xk = static_cast<float>(stars[idx].x);
            const float yk = static_cast<float>(stars[idx].y);

            const float vx = xk - xi;
            const float vy = yk - yi;
            const float dist = std::sqrt(vx * vx + vy * vy);
            if (dist <= 1e-6f) {
                valid = false;
                break;
            }

            all_rows.push_back(dist / image_diag);
            vectors.emplace_back(vx, vy);
        }

        if (!valid) {
            for (int rollback = 0; rollback < static_cast<int>(vectors.size()); rollback++) {
                all_rows.pop_back();
            }
            continue;
        }

        for (int k = 0; k < neighbors - 1; k++) {
            const cv::Point2f &v1 = vectors[k];
            const cv::Point2f &v2 = vectors[k + 1];
            const float d1 = std::sqrt(v1.x * v1.x + v1.y * v1.y);
            const float d2 = std::sqrt(v2.x * v2.x + v2.y * v2.y);
            float cos_angle = (v1.x * v2.x + v1.y * v2.y) / (d1 * d2);
            cos_angle = clamp(cos_angle, -1.0f, 1.0f);
            const float angle_norm = std::acos(cos_angle) / static_cast<float>(CV_PI);
            all_rows.push_back(angle_norm);
        }

        keypoints.emplace_back(cv::Point2f(xi, yi), 5.0f);
    }

    if (keypoints.size() < 4) {
        keypoints.clear();
        return false;
    }

    descriptors = cv::Mat(static_cast<int>(keypoints.size()), descriptor_dim, CV_32F);
    const int rows = descriptors.rows;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < descriptor_dim; c++) {
            descriptors.at<float>(r, c) = all_rows[static_cast<size_t>(r) * static_cast<size_t>(descriptor_dim) + static_cast<size_t>(c)];
        }
    }

    return true;
}

// Apply affine transform to planar RGB data (separate R, G, B planes)
// Avoids merge/split overhead by processing each plane independently
void warp_affine_planar(const u_int16_t *source, u_int16_t *dest,
                        const cv::Mat &affine_2x3, long width, long height,
                        int interpolation = cv::INTER_LINEAR,
                        int border_mode = cv::BORDER_CONSTANT,
                        u_int16_t border_value = 0) {
    // Create cv::Mat headers for each plane (no data copy)
    long npixels = width * height;
    cv::Mat source_r(height, width, CV_16UC1, const_cast<u_int16_t*>(source));
    cv::Mat source_g(height, width, CV_16UC1, const_cast<u_int16_t*>(source + npixels));
    cv::Mat source_b(height, width, CV_16UC1, const_cast<u_int16_t*>(source + 2*npixels));
    
    cv::Mat dest_r(height, width, CV_16UC1, dest);
    cv::Mat dest_g(height, width, CV_16UC1, dest + npixels);
    cv::Mat dest_b(height, width, CV_16UC1, dest + 2*npixels);
    
    // Apply same affine transform to each plane
    cv::warpAffine(source_r, dest_r, affine_2x3, cv::Size(width, height), interpolation, border_mode, cv::Scalar(border_value));
    cv::warpAffine(source_g, dest_g, affine_2x3, cv::Size(width, height), interpolation, border_mode, cv::Scalar(border_value));
    cv::warpAffine(source_b, dest_b, affine_2x3, cv::Size(width, height), interpolation, border_mode, cv::Scalar(border_value));
}

cv::Mat draw_star_boxes(const cv::Mat &base_gray, const star *stars, u_int32_t count) {
    cv::Mat vis;
    cv::cvtColor(base_gray, vis, cv::COLOR_GRAY2BGR);

    for (u_int32_t i = 0; i < count; i++) {
        const int x = static_cast<int>(stars[i].start_x);
        const int y = static_cast<int>(stars[i].start_y);
        const int w = static_cast<int>(stars[i].size_x);
        const int h = static_cast<int>(stars[i].size_y);
        cv::rectangle(vis, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 1);
    }

    return vis;
}

void show_debug_windows(const cv::Mat &gray1,
                        const cv::Mat &gray2,
                        const cv::Mat &thr1,
                        const cv::Mat &thr2,
                        const cv::Mat &stars1,
                        const cv::Mat &stars2,
                        const cv::Mat *matches) {
    cv::imshow("01 Gray Image 1", gray1);
    cv::imshow("02 Gray Image 2", gray2);
    cv::imshow("03 Threshold Image 1", thr1);
    cv::imshow("04 Threshold Image 2", thr2);
    cv::imshow("05 Detected Stars Image 1", stars1);
    cv::imshow("06 Detected Stars Image 2", stars2);
    if (matches != nullptr && !matches->empty()) {
        cv::imshow("07 Matched Stars", *matches);
    }

    printf("Debug windows opened. Press any key in an image window to continue...\n");
    cv::waitKey(0);
    cv::destroyAllWindows();
}

} // namespace

int main(int argc, char **argv) {
    char *filename1 = nullptr;
    char *filename2 = nullptr;

    int opt, option_index = 0;
    long num;
    char *end;

    threshold_params t_par;
        t_par.type = OTSU_CENTRALIZED;
        t_par.threshold = 1500;
        t_par.window_size = 201;
        t_par.reduce_factor = 8;
        t_par.threshold_scale = 0.8f;

    u_int16_t max_star_size = 100;
    u_int16_t min_star_size = 10;
    u_int16_t max_stars = 1000;
    int descriptor_neighbors = 2;
    bool show_steps = true;
    bool threshold_algo_set = false;

    static struct option long_options[] = {
        {"input-file1", required_argument, 0, 'f'},
        {"input-file2", required_argument, 0, 'g'},
        {"threshold", optional_argument, 0, 't'},
        {"reduce-factor", optional_argument, 0, 'r'},
        {"threshold-algorith", optional_argument, 0, 'a'},
        {"window-size", optional_argument, 0, 'w'},
        {"max-star-size", optional_argument, 0, 'M'},
        {"min-star-size", optional_argument, 0, 'm'},
        {"descriptor-neighbors", optional_argument, 0, 'k'},
        {"no-show", no_argument, 0, 'n'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "f:g:t:r:a:w:M:m:k:n", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'f':
                filename1 = optarg;
                break;
            case 'g':
                filename2 = optarg;
                break;
            case 't':
                num = strtol(optarg, &end, 10);
                if (end != optarg && num >= 0 && num <= 65535) {
                    t_par.threshold = static_cast<u_int16_t>(num);
                }
                break;
            case 'r':
                num = strtol(optarg, &end, 10);
                if (end != optarg && num >= 1 && num <= 65535) {
                    t_par.reduce_factor = static_cast<u_int16_t>(num);
                }
                break;
            case 'a':
                threshold_algo_set = true;
                if (strcmp(optarg, "simple") == 0) {
                    t_par.type = TR_SIMPLE;
                } else if (strcmp(optarg, "adaptive") == 0) {
                    t_par.type = TR_ADAPTIVE;
                } else if (strcmp(optarg, "fast-adaptive") == 0) {
                    t_par.type = TR_FAST_ADAPTIVE;
                }
                break;
            case 'w':
                num = strtol(optarg, &end, 10);
                if (end != optarg && num >= 1 && num <= 65535) {
                    t_par.window_size = static_cast<u_int16_t>(num);
                }
                break;
            case 'M':
                num = strtol(optarg, &end, 10);
                if (end != optarg && num >= 1 && num <= 65535) {
                    max_star_size = static_cast<u_int16_t>(num);
                }
                break;
            case 'm':
                num = strtol(optarg, &end, 10);
                if (end != optarg && num >= 0 && num <= 65535 && num < max_star_size) {
                    min_star_size = static_cast<u_int16_t>(num);
                }
                break;
            case 'k':
                num = strtol(optarg, &end, 10);
                if (end != optarg && num >= 2 && num <= 64) {
                    descriptor_neighbors = static_cast<int>(num);
                }
                break;
            case 'n':
                show_steps = false;
                break;
            default:
                fprintf(stderr, "Usage: %s --input-file1 <image1.fits> --input-file2 <image2.fits>\n", argv[0]);
                return 1;
        }
    }

    if (filename1 == nullptr || filename2 == nullptr) {
        fprintf(stderr, "Usage: %s --input-file1 <image1.fits> --input-file2 <image2.fits>\n", argv[0]);
        return 1;
    }

    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    CHECK(cudaSetDevice(dev));
    PrefetchDeviceArg devLoc = make_prefetch_device_arg(dev);

    fitsfile *fptr1 = nullptr;
    fitsfile *fptr2 = nullptr;
    long width1, height1, channels1;
    long width2, height2, channels2;
    open_fits(filename1, &fptr1);
    open_fits(filename2, &fptr2);
    get_fits_dimensions(fptr1, &width1, &height1, &channels1);
    get_fits_dimensions(fptr2, &width2, &height2, &channels2);

    if (width1 <= 0 || height1 <= 0 || channels1 <= 0 || width2 <= 0 || height2 <= 0 || channels2 <= 0) {
        fprintf(stderr, "Invalid FITS dimensions\n");
        return 1;
    }

    if (width1 != width2 || height1 != height2 || channels1 != channels2) {
        fprintf(stderr,
                "Input images must have same width/height/channels. image1=(%ld,%ld,%ld), image2=(%ld,%ld,%ld)\n",
                width1, height1, channels1, width2, height2, channels2);
        return 1;
    }

    if (channels1 != 1 && channels1 != 3) {
        fprintf(stderr, "Unsupported channels: %ld. Supported values are 1 or 3.\n", channels1);
        return 1;
    }

    if (channels1 == 1 && !threshold_algo_set && t_par.type == TR_SIMPLE) {
        t_par.type = TR_ADAPTIVE;
        printf("Mono input detected: switching default threshold mode from simple to adaptive.\n");
    }

    const long width = width1;
    const long height = height1;
    const long channels = channels1;

    const u_int64_t npixels = static_cast<u_int64_t>(width) * static_cast<u_int64_t>(height);
    const u_int64_t totpixels = npixels * static_cast<u_int64_t>(channels);

    u_int16_t *fits_data1 = nullptr;
    u_int16_t *fits_data2 = nullptr;
    CHECK(cudaMallocManaged(&fits_data1, totpixels * sizeof(u_int16_t)));
    CHECK(cudaMallocManaged(&fits_data2, totpixels * sizeof(u_int16_t)));

    u_int16_t *gray_image1 = nullptr;
    u_int16_t *gray_image2 = nullptr;
    CHECK(cudaMallocManaged(&gray_image1, npixels * sizeof(u_int16_t)));
    CHECK(cudaMallocManaged(&gray_image2, npixels * sizeof(u_int16_t)));
    CHECK(cudaMemPrefetchAsync(gray_image1, npixels * sizeof(u_int16_t), devLoc, 0));
    CHECK(cudaMemPrefetchAsync(gray_image2, npixels * sizeof(u_int16_t), devLoc, 0));

    u_int8_t *threshold_image1 = nullptr;
    u_int8_t *threshold_image2 = nullptr;
    CHECK(cudaMallocManaged(&threshold_image1, npixels * sizeof(u_int8_t)));
    CHECK(cudaMallocManaged(&threshold_image2, npixels * sizeof(u_int8_t)));
    CHECK(cudaMemPrefetchAsync(threshold_image1, npixels * sizeof(u_int8_t), devLoc, 0));
    CHECK(cudaMemPrefetchAsync(threshold_image2, npixels * sizeof(u_int8_t), devLoc, 0));

    get_fits_data(fptr1, totpixels, fits_data1);
    get_fits_data(fptr2, totpixels, fits_data2);
    CHECK(cudaMemPrefetchAsync(fits_data1, totpixels * sizeof(u_int16_t), devLoc, 0));
    CHECK(cudaMemPrefetchAsync(fits_data2, totpixels * sizeof(u_int16_t), devLoc, 0));

    u_int16_t *rgb_data1 = fits_data1;
    u_int16_t *rgb_data2 = fits_data2;

    double t_start = cpuSecond();

    if (channels == 1) {
        CHECK(cudaMallocManaged(&rgb_data1, npixels * 3 * sizeof(u_int16_t)));
        CHECK(cudaMallocManaged(&rgb_data2, npixels * 3 * sizeof(u_int16_t)));
        CHECK(cudaMemPrefetchAsync(rgb_data1, npixels * 3 * sizeof(u_int16_t), devLoc, 0));
        CHECK(cudaMemPrefetchAsync(rgb_data2, npixels * 3 * sizeof(u_int16_t), devLoc, 0));

        // For mono input, debayer first to obtain RGB planes expected by the rest of pipeline.
        demosaic_bilinear_rggb_cpu(fits_data1, rgb_data1, width, height, 1);
        demosaic_bilinear_rggb_cpu(fits_data2, rgb_data2, width, height, 1);
    }

    to_grayscale_planar_gpu(rgb_data1, gray_image1, npixels);
    to_grayscale_planar_gpu(rgb_data2, gray_image2, npixels);

    double t_elapsed = cpuSecond() - t_start;
    printf("Grayscale done - time: %f\n", t_elapsed);
    print_u16_stats("Gray1 stats", gray_image1, npixels);
    print_u16_stats("Gray2 stats", gray_image2, npixels);

    t_start = cpuSecond();
    compute_threshold_gpu(gray_image1, threshold_image1, width, height, t_par);
    compute_threshold_gpu(gray_image2, threshold_image2, width, height, t_par);
    t_elapsed = cpuSecond() - t_start;
    printf("Threshold done - time: %f\n", t_elapsed);
    print_u8_threshold_stats("Threshold1 stats", threshold_image1, npixels);
    print_u8_threshold_stats("Threshold2 stats", threshold_image2, npixels);

    star *d_stars1 = nullptr;
    star *d_stars2 = nullptr;
    CHECK(cudaMallocManaged(&d_stars1, max_stars * sizeof(star)));
    CHECK(cudaMallocManaged(&d_stars2, max_stars * sizeof(star)));

    u_int32_t *d_num_stars1 = nullptr;
    u_int32_t *d_num_stars2 = nullptr;
    CHECK(cudaMallocManaged(&d_num_stars1, sizeof(u_int32_t)));
    CHECK(cudaMallocManaged(&d_num_stars2, sizeof(u_int32_t)));
    *d_num_stars1 = 0;
    *d_num_stars2 = 0;

    t_start = cpuSecond();
    detect_stars_gpu(threshold_image1, width, height, max_star_size, min_star_size, d_stars1, d_num_stars1, max_stars);
    detect_stars_gpu(threshold_image2, width, height, max_star_size, min_star_size, d_stars2, d_num_stars2, max_stars);
    t_elapsed = cpuSecond() - t_start;
    printf("Star detection done - time: %f\n", t_elapsed);

    CHECK(cudaDeviceSynchronize());

    cv::Mat img1_u8 = normalize_u16_to_u8(gray_image1, static_cast<int>(width), static_cast<int>(height));
    cv::Mat img2_u8 = normalize_u16_to_u8(gray_image2, static_cast<int>(width), static_cast<int>(height));
    cv::Mat threshold1(static_cast<int>(height), static_cast<int>(width), CV_8UC1, (void*)threshold_image1);
    cv::Mat threshold2(static_cast<int>(height), static_cast<int>(width), CV_8UC1, (void*)threshold_image2);
    cv::Mat stars_vis1 = draw_star_boxes(img1_u8, d_stars1, *d_num_stars1);
    cv::Mat stars_vis2 = draw_star_boxes(img2_u8, d_stars2, *d_num_stars2);

    if (*d_num_stars1 == 0 || *d_num_stars2 == 0) {
        fprintf(stderr, "No stars detected in one or both images (image1=%u, image2=%u).\n", *d_num_stars1, *d_num_stars2);
        if (show_steps) {
            show_debug_windows(img1_u8, img2_u8, threshold1, threshold2, stars_vis1, stars_vis2, nullptr);
        }
        return 1;
    }

    star_detail *stars_details1 = nullptr;
    star_detail *stars_details2 = nullptr;
    CHECK(cudaMallocManaged(&stars_details1, (*d_num_stars1) * sizeof(star_detail)));
    CHECK(cudaMallocManaged(&stars_details2, (*d_num_stars2) * sizeof(star_detail)));

    t_start = cpuSecond();
    populate_star_details_gpu(stars_details1, d_stars1, *d_num_stars1, rgb_data1, gray_image1, width, npixels);
    populate_star_details_gpu(stars_details2, d_stars2, *d_num_stars2, rgb_data2, gray_image2, width, npixels);
    t_elapsed = cpuSecond() - t_start;
    printf("Star details population done - time: %f\n", t_elapsed);

    CHECK(cudaDeviceSynchronize());

    printf("Detected %u stars in image1\n", *d_num_stars1);
    printf("Detected %u stars in image2\n", *d_num_stars2);
    printf("Descriptor neighbors: %d\n", descriptor_neighbors);

    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
    cv::Mat descriptors1, descriptors2;

    bool ok1 = false;
    bool ok2 = false;
    if (descriptor_neighbors == 2) {
        ok1 = build_star_descriptors(stars_details1, *d_num_stars1, width, height, keypoints1, descriptors1);
        ok2 = build_star_descriptors(stars_details2, *d_num_stars2, width, height, keypoints2, descriptors2);
    } else {
        ok1 = build_star_descriptors_generalized(stars_details1, *d_num_stars1, width, height, descriptor_neighbors, keypoints1, descriptors1);
        ok2 = build_star_descriptors_generalized(stars_details2, *d_num_stars2, width, height, descriptor_neighbors, keypoints2, descriptors2);
    }

    if (!ok1 || !ok2) {
        fprintf(stderr,
                "Not enough stars to build descriptors (need at least 4 valid stars and >= k+1 stars per image, k=%d).\n",
                descriptor_neighbors);
        return 1;
    }

    cv::BFMatcher matcher(cv::NORM_L2, false);
    const float ratio_thresh = 0.7f;

    std::vector<std::vector<cv::DMatch>> knn_12;
    std::vector<std::vector<cv::DMatch>> knn_21;
    matcher.knnMatch(descriptors1, descriptors2, knn_12, 2);
    matcher.knnMatch(descriptors2, descriptors1, knn_21, 2);

    std::vector<cv::DMatch> forward_ratio_matches;
    for (const auto &m : knn_12) {
        if (m.size() >= 2 && m[0].distance < ratio_thresh * m[1].distance) {
            forward_ratio_matches.push_back(m[0]);
        }
    }

    std::vector<int> reverse_best_query(descriptors2.rows, -1);
    for (const auto &m : knn_21) {
        if (m.size() >= 2 && m[0].distance < ratio_thresh * m[1].distance) {
            // reverse match: queryIdx is in image2 descriptor set, trainIdx is in image1 descriptor set
            reverse_best_query[m[0].queryIdx] = m[0].trainIdx;
        }
    }

    std::vector<cv::DMatch> mutual_matches;
    mutual_matches.reserve(forward_ratio_matches.size());
    for (const auto &m : forward_ratio_matches) {
        if (m.trainIdx >= 0 && m.trainIdx < static_cast<int>(reverse_best_query.size())
            && reverse_best_query[m.trainIdx] == m.queryIdx) {
            mutual_matches.push_back(m);
        }
    }

    std::vector<cv::DMatch> inlier_matches;
    cv::Mat affine_2x3;
    if (mutual_matches.size() >= 3) {
        std::vector<cv::Point2f> points1, points2;
        points1.reserve(mutual_matches.size());
        points2.reserve(mutual_matches.size());

        for (const auto &m : mutual_matches) {
            points1.push_back(keypoints1[m.queryIdx].pt);
            points2.push_back(keypoints2[m.trainIdx].pt);
        }

        cv::Mat inlier_mask;
        affine_2x3 = cv::estimateAffinePartial2D(
            points2,
            points1,
            inlier_mask,
            cv::RANSAC,
            3.0,
            2000,
            0.99,
            10
        );

        if (!affine_2x3.empty() && !inlier_mask.empty()) {
            inlier_matches.reserve(mutual_matches.size());
            for (int i = 0; i < inlier_mask.rows; i++) {
                if (inlier_mask.at<uchar>(i, 0)) {
                    inlier_matches.push_back(mutual_matches[static_cast<size_t>(i)]);
                }
            }
        }
    }

    printf("Built %d descriptors for image1 and %d for image2\n", descriptors1.rows, descriptors2.rows);
    printf("Forward ratio matches: %zu\n", forward_ratio_matches.size());
    printf("Mutual symmetric matches: %zu\n", mutual_matches.size());
    printf("RANSAC affine inlier matches: %zu\n", inlier_matches.size());

    cv::Mat match_img;
    cv::drawMatches(
        img1_u8,
        keypoints1,
        img2_u8,
        keypoints2,
        inlier_matches,
        match_img,
        cv::Scalar::all(-1),
        cv::Scalar::all(-1),
        std::vector<char>(),
        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    const std::string out_dir = "output_star";
    std::error_code fs_err;
    //std::filesystem::create_directories(out_dir, fs_err);

    const std::string match_out_path = out_dir + "/star_matches.png";
    if (!cv::imwrite(match_out_path, match_img)) {
        fprintf(stderr, "Failed to write match image %s\n", match_out_path.c_str());
    } else {
        printf("Saved matches image: %s\n", match_out_path.c_str());
    }

    if (!affine_2x3.empty() && inlier_matches.size() >= 3) {
            // Allocate output buffers for aligned RGB planar data
            u_int16_t *aligned = nullptr;
            CHECK(cudaMallocManaged(&aligned, npixels * sizeof(u_int16_t) * 3));
            
            warp_affine_planar(rgb_data2, aligned, affine_2x3, width, height);
            
            save_image_fits(out_dir, "aligned_image2", aligned, width, height, 3);
            CHECK(cudaFree(aligned));
    } else {
        fprintf(stderr, "Not enough inlier matches to compute affine transform (need at least 3).\n");
    }

    if (show_steps) {
        show_debug_windows(img1_u8, img2_u8, threshold1, threshold2, stars_vis1, stars_vis2, &match_img);
    }

    return 0;
}
