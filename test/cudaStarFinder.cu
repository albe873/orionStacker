#include "cuda_runtime.h"

#include "fits_helper.h"
#include "cuda_helper.h"
#include "common.h"

#include "star_finder.h"

#include <stdio.h>
#include <getopt.h>
#include <unistd.h>
#include <cstring>

#include "opencv2/imgcodecs.hpp"
#include <cstdlib>

int main(int argc, char **argv) {

    char *filename = nullptr;
    int opt, option_index = 0;
    long num;
    char *end;
    threshold_params t_par;
        t_par.type = TR_SIMPLE;
        t_par.threshold = 1500;
        t_par.window_size = 255;
        t_par.reduce_factor = 8;

    u_int16_t max_star_size = 100;
    u_int16_t min_star_size = 10;
    u_int16_t max_stars = 1000;

    static struct option long_options[] = {
        {"input-file", required_argument, 0, 'f'},
        {"threshold", optional_argument, 0, 't'},
        {"reduce-factor", optional_argument, 0, 'r'},
        {"threshold-algorith", optional_argument, 0, 'a'},
        {"window-size", optional_argument, 0, 'w'},
        {"max-star-size", optional_argument, 0, 'M'},
        {"min-star-size", optional_argument, 0, 'm'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "f:t:r:a:w:M:m:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'f':
                filename = optarg;
                break;
            case 't':
                num = strtol(optarg, &end, 10);
                if (end == optarg) {
                    fprintf(stderr, "Cannot convert threshold value, using default\n");
                } else if (num < 0 || num > 65535) {
                    fprintf(stderr, "Invalid threshold value, using default\n");
                } else {
                    t_par.threshold = num;
                }
                break;
            case 'r':
                num = strtol(optarg, &end, 10);
                if (end == optarg) {
                    fprintf(stderr, "Cannot convert reduce factor, using default\n");
                } else if (num < 1 || num > 65535) {
                    fprintf(stderr, "Invalid reduce factor, using default\n");
                } else {
                    t_par.reduce_factor = num;
                }
                break;
            case 'a':
                if (strcmp(optarg, "simple") == 0) {
                    t_par.type = TR_SIMPLE;
                } else if (strcmp(optarg, "adaptive") == 0) {
                    t_par.type = TR_ADAPTIVE;
                } else if (strcmp(optarg, "fast-adaptive") == 0) {
                    t_par.type = TR_FAST_ADAPTIVE;
                } else {
                    fprintf(stderr, "Invalid threshold algorithm, using default\n");
                }
                break;
            case 'w':
                num = strtol(optarg, &end, 10);
                if (end == optarg) {
                    fprintf(stderr, "Cannot convert window size, using default\n");
                } else if (num < 1 || num > 65535) {
                    fprintf(stderr, "Invalid window size, using default\n");
                } else {
                    t_par.window_size = num;
                }
                break;
            case 'M':
                num = strtol(optarg, &end, 10);
                if (end == optarg) {
                    fprintf(stderr, "Cannot convert max star size, using default\n");
                } else if (num < 1 || num > 65535) {
                    fprintf(stderr, "Invalid max star size, using default\n");
                } else {
                    max_star_size = num;
                }
                break;
            case 'm':
                num = strtol(optarg, &end, 10);
                if (end == optarg) {
                    fprintf(stderr, "Cannot convert min star size, using default\n");
                } else if (num < 0 || num > 65535 || num >= max_star_size) {
                    fprintf(stderr, "Invalid min star size, using default\n");
                } else {
                    min_star_size = num;
                }
                break;
            default:
                fprintf(stderr, "Usage: %s --input-file <image.fits>\n", argv[0]);
                return 1;
        }
    }

    if (filename == nullptr) {
        fprintf(stderr, "Usage: %s --input-file <image.fits>\n", argv[0]);
        return 1;
    }

    // Inizializza CUDA
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    CHECK(cudaSetDevice(dev));
    PrefetchDeviceArg devLoc = make_prefetch_device_arg(dev);

    // --- Apre il file FITS ---
    fitsfile *fptr = nullptr;
    long width, height, channels;
    open_fits(filename, &fptr);
    get_fits_dimensions(fptr, &width, &height, &channels);

    if (width * height * channels == 0) {
        fprintf(stderr, "Invalid image dimensions\n");
        return 1;
    }

    // Alloca memoria per l'immagine FITS e versione grayscale
    u_int64_t totpixels = width * height * channels;
    u_int64_t npixels = width * height;

    u_int16_t *fits_data = nullptr;
    CHECK(cudaMallocManaged(&fits_data, totpixels * sizeof(u_int16_t)));

    u_int16_t *gray_image = nullptr;
    CHECK(cudaMallocManaged(&gray_image, npixels * sizeof(u_int16_t)));
    CHECK(cudaMemPrefetchAsync(gray_image, npixels * sizeof(u_int16_t), devLoc, 0));

    u_int8_t *threshold_image = nullptr;
    CHECK(cudaMallocManaged(&threshold_image, npixels * sizeof(u_int8_t)));
    CHECK(cudaMemPrefetchAsync(threshold_image, npixels * sizeof(u_int8_t), devLoc, 0));

    // Legge i dati dal file FITS
    get_fits_data(fptr, totpixels, fits_data);
    CHECK(cudaMemPrefetchAsync(fits_data, totpixels * sizeof(u_int16_t), devLoc, 0));

    double t_start, t_elapsed;
    t_start = cpuSecond();

    // --- Convert to grayscale ---
    to_grayscale_planar_gpu(fits_data, gray_image, npixels);
    t_elapsed = cpuSecond() - t_start;
    printf("Grayscale done - time: %f\n", t_elapsed);
    t_start = cpuSecond();

    // --- Compute threshold ---
    compute_threshold_gpu(gray_image, threshold_image, width, height, t_par);
    t_elapsed = cpuSecond() - t_start;
    printf("Threshold done - time: %f\n", t_elapsed);

    // --- Detect stars ---

    // Allocate memory for the stars info
    star *d_stars = nullptr;
    CHECK(cudaMallocManaged(&d_stars, max_stars * sizeof(star)));

    u_int32_t *d_num_stars = nullptr;
    CHECK(cudaMallocManaged(&d_num_stars, sizeof(u_int32_t)));
    *d_num_stars = 0;   // initialized to 0
    
    // Detect
    t_start = cpuSecond();
    detect_stars_gpu(threshold_image, width, height, max_star_size, min_star_size, d_stars, d_num_stars, max_stars);
    t_elapsed = cpuSecond() - t_start;
    printf("Star detection done - time: %f\n", t_elapsed);

    // write threshold image to disk
    const char *threshold_dir = "output_gray";

    cv::Mat img((int)height, (int)width, CV_8UC1, (void*)threshold_image);

    std::string out_path = std::string(threshold_dir) + "/threshold.png";
    if (!cv::imwrite(out_path, img)) {
        fprintf(stderr, "Failed to write PNG %s\n", out_path.c_str());
    } else {
        printf("Saved PNG %s\n", out_path.c_str());
    }

    // Get stars details
    star_detail *stars_details = nullptr;
    printf("Allocating stars details for %u stars\n", *d_num_stars);
    CHECK(cudaMallocManaged(&stars_details, *d_num_stars * sizeof(star_detail)));
    printf("Allocated stars details\n");

    t_start = cpuSecond();
    populate_star_details_gpu(stars_details, d_stars, *d_num_stars, fits_data, gray_image, width, npixels);
    t_elapsed = cpuSecond() - t_start;
    printf("Star details population done - time: %f\n", t_elapsed);

    // --- Print the stars info ---
    printf("Detected %u stars:\n", *d_num_stars);
    for (u_int32_t i = 0; i < *d_num_stars; i++) {
        printf("Star %u: start=(%lu, %lu), size=(%u, %u)\n", i, d_stars[i].start_x, height - d_stars[i].start_y, d_stars[i].size_x, d_stars[i].size_y);
        printf("    Baricenter: (%.2f, %.2f), Brightness: R=%.2f G=%.2f B=%.2f Total=%.2f\n", 
               stars_details[i].x, height - stars_details[i].y, 
               stars_details[i].b_red, stars_details[i].b_green, stars_details[i].b_blue, stars_details[i].b);
    }


    // --- Draw stars on the image ---
    draw_stars(fits_data, width, d_stars, *d_num_stars);

    // --- Save images ---
    const char *detect_dir = "output_star";
    save_image_fits(detect_dir, "detect_output", fits_data, width, height, channels);


    // Ensure CUDA work is finished so unified memory is ready on host
    CHECK(cudaDeviceSynchronize());

}
