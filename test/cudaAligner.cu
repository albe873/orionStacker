#include "fits_helper.hh"
#include "cuda_helper.hh"
#include "common.hh"

#include "star_finder.hh"
#include "debayer.hh"

#include <stdio.h>
#include <getopt.h>
#include <unistd.h>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"

namespace {

cv::Mat normalize_u16_to_u8(const uint16_t *img, int width, int height) {
    cv::Mat src_u16(height, width, CV_16UC1, const_cast<u_int16_t*>(img));
    cv::Mat out_u8;
    cv::normalize(src_u16, out_u8, 0, 255, cv::NORM_MINMAX, CV_8U);
    return out_u8;
}

void print_u16_stats(const char *label, const uint16_t *img, uint64_t npixels) {
    if (npixels == 0) {
        printf("%s: empty\n", label);
        return;
    }

    uint16_t min_v = std::numeric_limits<u_int16_t>::max();
    uint16_t max_v = 0;
    uint64_t sum = 0;
    for (uint64_t i = 0; i < npixels; i++) {
        const uint16_t v = img[i];
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
        sum += static_cast<uint64_t>(v);
    }
    const double mean = static_cast<double>(sum) / static_cast<double>(npixels);
    printf("%s: min=%u max=%u mean=%.2f\n", label, static_cast<unsigned>(min_v), static_cast<unsigned>(max_v), mean);
}


// ---------------------------------------------------------------------------
// Unified wrapper – picks the GPU path by default.
// ---------------------------------------------------------------------------
void warp_affine_planar(const uint16_t *source, uint16_t *dest,
                        const cv::Mat &affine_2x3, long width, long height) {
    warp_affine_planar_gpu(source, dest, affine_2x3, width, height);
}

cv::Mat draw_star_boxes(const cv::Mat &base_gray, const star *stars, uint32_t count) {
    cv::Mat vis;
    cv::cvtColor(base_gray, vis, cv::COLOR_GRAY2BGR);

    for (uint32_t i = 0; i < count; i++) {
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
        t_par.window_size = 201;

    uint16_t max_star_size = 100;
    uint16_t min_star_size = 10;
    uint16_t max_stars = 1000;
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
                    t_par.threshold = static_cast<uint16_t>(num);
                }
                break;
            case 'r':
                num = strtol(optarg, &end, 10);
                if (end != optarg && num >= 1 && num <= 65535) {
                    t_par.reduce_factor = static_cast<uint16_t>(num);
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
                    t_par.window_size = static_cast<uint16_t>(num);
                }
                break;
            case 'M':
                num = strtol(optarg, &end, 10);
                if (end != optarg && num >= 1 && num <= 65535) {
                    max_star_size = static_cast<uint16_t>(num);
                }
                break;
            case 'm':
                num = strtol(optarg, &end, 10);
                if (end != optarg && num >= 0 && num <= 65535 && num < max_star_size) {
                    min_star_size = static_cast<uint16_t>(num);
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

    const long width = width1;
    const long height = height1;
    const long channels = channels1;

    const uint64_t npixels = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
    const uint64_t totpixels = npixels * static_cast<uint64_t>(channels);

    uint16_t *fits_data1 = nullptr;
    uint16_t *fits_data2 = nullptr;
    CHECK(cudaMallocManaged(&fits_data1, totpixels * sizeof(uint16_t)));
    CHECK(cudaMallocManaged(&fits_data2, totpixels * sizeof(uint16_t)));

    uint16_t *gray_image1 = nullptr;
    uint16_t *gray_image2 = nullptr;
    CHECK(cudaMallocManaged(&gray_image1, npixels * sizeof(uint16_t)));
    CHECK(cudaMallocManaged(&gray_image2, npixels * sizeof(uint16_t)));
    CHECK(cudaMemPrefetchAsync(gray_image1, npixels * sizeof(uint16_t), devLoc, 0));
    CHECK(cudaMemPrefetchAsync(gray_image2, npixels * sizeof(uint16_t), devLoc, 0));

    uint8_t *threshold_image1 = nullptr;
    uint8_t *threshold_image2 = nullptr;
    CHECK(cudaMallocManaged(&threshold_image1, npixels * sizeof(uint8_t)));
    CHECK(cudaMallocManaged(&threshold_image2, npixels * sizeof(uint8_t)));
    CHECK(cudaMemPrefetchAsync(threshold_image1, npixels * sizeof(uint8_t), devLoc, 0));
    CHECK(cudaMemPrefetchAsync(threshold_image2, npixels * sizeof(uint8_t), devLoc, 0));

    get_fits_data(fptr1, totpixels, fits_data1);
    get_fits_data(fptr2, totpixels, fits_data2);
    CHECK(cudaMemPrefetchAsync(fits_data1, totpixels * sizeof(uint16_t), devLoc, 0));
    CHECK(cudaMemPrefetchAsync(fits_data2, totpixels * sizeof(uint16_t), devLoc, 0));

    uint16_t *rgb_data1 = fits_data1;
    uint16_t *rgb_data2 = fits_data2;

    double t_start = cpuSecond();

    if (channels == 1) {
        CHECK(cudaMallocManaged(&rgb_data1, npixels * 3 * sizeof(uint16_t)));
        CHECK(cudaMallocManaged(&rgb_data2, npixels * 3 * sizeof(uint16_t)));
        CHECK(cudaMemPrefetchAsync(rgb_data1, npixels * 3 * sizeof(uint16_t), devLoc, 0));
        CHECK(cudaMemPrefetchAsync(rgb_data2, npixels * 3 * sizeof(uint16_t), devLoc, 0));

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

    star *d_stars1 = nullptr;
    star *d_stars2 = nullptr;
    CHECK(cudaMallocManaged(&d_stars1, max_stars * sizeof(star)));
    CHECK(cudaMallocManaged(&d_stars2, max_stars * sizeof(star)));

    uint32_t *d_num_stars1 = nullptr;
    uint32_t *d_num_stars2 = nullptr;
    CHECK(cudaMallocManaged(&d_num_stars1, sizeof(uint32_t)));
    CHECK(cudaMallocManaged(&d_num_stars2, sizeof(uint32_t)));
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

    std::vector<cv::DMatch> inlier_matches;
    cv::Mat affine_2x3;
    float ratio_threshold = 0.7F; 
    affine_2x3 = estimate_affine_partial_stars(keypoints1, descriptors1, keypoints2, descriptors2, ratio_threshold, &inlier_matches);

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

        // ---------- CPU reference ----------
        uint16_t *aligned_cpu = nullptr;
        CHECK(cudaMallocManaged(&aligned_cpu, npixels * sizeof(u_int16_t) * 3));

        double t_cpu = cpuSecond();
        warp_affine_planar_cpu(rgb_data2, aligned_cpu, affine_2x3, width, height);
        t_cpu = cpuSecond() - t_cpu;

        // ---------- GPU ----------
        uint16_t *aligned_gpu = nullptr;
        CHECK(cudaMallocManaged(&aligned_gpu, npixels * sizeof(u_int16_t) * 3));
        // Prefetch GPU destination buffer to device to avoid first-touch page faults
        CHECK(cudaMemPrefetchAsync(aligned_gpu, npixels * sizeof(u_int16_t) * 3, devLoc, 0));

        double t_gpu = cpuSecond();
        warp_affine_planar_gpu(rgb_data2, aligned_gpu, affine_2x3, width, height);
        t_gpu = cpuSecond() - t_gpu;

        // ---------- Pixel-difference statistics ----------
        uint64_t diff_pixels = 0;
        uint64_t diff_sum    = 0;
        uint16_t diff_max    = 0;
        for (uint64_t i = 0; i < npixels * 3; i++) {
            auto cpu_v = aligned_cpu[i];
            auto gpu_v = aligned_gpu[i];
            if (cpu_v != gpu_v) {
                diff_pixels++;
                uint16_t d = (cpu_v > gpu_v) ? (uint16_t)(cpu_v - gpu_v) : (uint16_t)(gpu_v - cpu_v);
                diff_sum += d;
                if (d > diff_max) diff_max = d;
            }
        }
        double diff_pct = 100.0 * (double)diff_pixels / (double)(npixels * 3);

        printf("\nTiming:\n");
        printf("  CPU warpAffine:  %.3f ms\n", t_cpu * 1000.0);
        printf("  GPU warp kernel: %.3f ms\n", t_gpu * 1000.0);
        printf("  Speedup:         %.2fx\n", t_cpu / t_gpu);

        printf("\nCPU vs GPU difference:\n");
        printf("  Different pixels: %llu (%.4f%%)\n", (unsigned long long)diff_pixels, diff_pct);
        printf("  Max difference:   %u\n", (unsigned)diff_max);
        if (diff_pixels > 0) {
            printf("  Mean difference:  %.2f\n", (double)diff_sum / (double)diff_pixels);
        }

        // Save both outputs for visual inspection
        save_image_fits(out_dir, "aligned_image2_cpu", aligned_cpu, width, height, 3);
        save_image_fits(out_dir, "aligned_image2_gpu", aligned_gpu, width, height, 3);

        CHECK(cudaFree(aligned_cpu));
        CHECK(cudaFree(aligned_gpu));
    } else {
        fprintf(stderr, "Not enough inlier matches to compute affine transform (need at least 3).\n");
    }

    if (show_steps) {
        show_debug_windows(img1_u8, img2_u8, threshold1, threshold2, stars_vis1, stars_vis2, &match_img);
    }

    return 0;
}
