#include "fits_helper.hh"
#include "cuda_helper.hh"
#include "common.hh"

#include "star_finder.hh"

#include <stdio.h>
#include <getopt.h>
#include <unistd.h>
#include <cstring>

#include "opencv2/imgcodecs.hpp"
#include <algorithm>

int main(int argc, char **argv) {

    char *filename = nullptr;
    int opt, option_index = 0;
    long num;
    char *end;
    threshold_params t_par;
        t_par.window_size = 201;

    uint16_t max_star_size = 100;
    uint16_t min_star_size = 4;
    uint16_t max_stars = 1000;

    static struct option long_options[] = {
        {"input-file", required_argument, 0, 'f'},
        {"threshold", optional_argument, 0, 't'},
        {"reduce-factor", optional_argument, 0, 'r'},
        {"threshold-algorithm", optional_argument, 0, 'a'},
        {"window-size", optional_argument, 0, 'w'},
        {"threshold-factor", optional_argument, 0, 'T'},
        {"max-star-size", optional_argument, 0, 'M'},
        {"min-star-size", optional_argument, 0, 'm'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "f:t:r:a:w:T:M:m:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'f':
                filename = optarg;
                break;
            case 't': {
                auto num = strtol(optarg, &end, 10);
                if (end == optarg) {
                    fprintf(stderr, "Cannot convert threshold value, using default\n");
                } else if (num < 0 || num > 65535) {
                    fprintf(stderr, "Invalid threshold value, using default\n");
                } else {
                    t_par.threshold = num;
                }
                break;}
            case 'r': {
                auto num = strtol(optarg, &end, 10);
                if (end == optarg) {
                    fprintf(stderr, "Cannot convert reduce factor, using default\n");
                } else if (num < 1 || num > 65535) {
                    fprintf(stderr, "Invalid reduce factor, using default\n");
                } else {
                    t_par.reduce_factor = num;
                }
                break;}
            case 'a': {
                if (strcmp(optarg, "simple") == 0) {
                    t_par.type = TR_SIMPLE;
                } else if (strcmp(optarg, "adaptive") == 0) {
                    t_par.type = TR_ADAPTIVE;
                } else if (strcmp(optarg, "fast-adaptive") == 0) {
                    t_par.type = TR_FAST_ADAPTIVE;
                } else if (strcmp(optarg, "otsu") == 0 || strcmp(optarg, "otsu-centralized") == 0) {
                    t_par.type = OTSU_CENTRALIZED;
                } else {
                    fprintf(stderr, "Invalid threshold algorithm, using default\n");
                }
                break;}
            case 'w': {
                auto num = strtol(optarg, &end, 10);
                if (end == optarg) {
                    fprintf(stderr, "Cannot convert window size, using default\n");
                } else if (num < 1 || num > 65535) {
                    fprintf(stderr, "Invalid window size, using default\n");
                } else {
                    t_par.window_size = num;
                }
                break;}
            case 'M': {
                auto num = strtol(optarg, &end, 10);
                if (end == optarg) {
                    fprintf(stderr, "Cannot convert max star size, using default\n");
                } else if (num < 1 || num > 65535) {
                    fprintf(stderr, "Invalid max star size, using default\n");
                } else {
                    max_star_size = num;
                }
                break;}
            case 'm': {
                auto num = strtol(optarg, &end, 10);
                if (end == optarg) {
                    fprintf(stderr, "Cannot convert min star size, using default\n");
                } else if (num < 0 || num > 65535 || num >= max_star_size) {
                    fprintf(stderr, "Invalid min star size, using default\n");
                } else {
                    min_star_size = num;
                }
                break;}
            case 'T': {
                float threshold_scale = strtof(optarg, &end);
                if (end == optarg) {
                    fprintf(stderr, "Cannot convert threshold scale, using default\n");
                } else if (threshold_scale <= 0.0f || threshold_scale > 10.0f) {
                    fprintf(stderr, "Invalid threshold scale, using default\n");
                } else {
                    printf("changed threshold scale to %f\n", threshold_scale);
                    t_par.threshold_scale = threshold_scale;
                }
                break;}
            default:
                fprintf(stderr, "Invalid argument: %c\n", opt);
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
    uint64_t totpixels = width * height * channels;
    uint64_t npixels = width * height;

    uint16_t *fits_data = nullptr;
    CHECK(cudaMallocManaged(&fits_data, totpixels * sizeof(uint16_t)));

    uint16_t *gray_image = nullptr;
    CHECK(cudaMallocManaged(&gray_image, npixels * sizeof(uint16_t)));
    CHECK(cudaMemPrefetchAsync(gray_image, npixels * sizeof(uint16_t), devLoc, 0));

    uint8_t *threshold_image = nullptr;
    CHECK(cudaMallocManaged(&threshold_image, npixels * sizeof(uint8_t)));
    CHECK(cudaMemPrefetchAsync(threshold_image, npixels * sizeof(uint8_t), devLoc, 0));

    // Legge i dati dal file FITS
    get_fits_data(fptr, totpixels, fits_data);
    CHECK(cudaMemPrefetchAsync(fits_data, totpixels * sizeof(uint16_t), devLoc, 0));
    CHECK(cudaDeviceSynchronize());

    // ======================================================
    // GPU part
    printf("\n======================================\n");
    printf("Running star detection on GPU\n\n");

    double t_start;
    t_start = cpuSecond();

    // 1 - Convert to grayscale
    to_grayscale_planar_gpu(fits_data, gray_image, npixels);
    double time_grayscale_gpu = cpuSecond() - t_start;
    printf("  Grayscale done - time: %f\n", time_grayscale_gpu);
    t_start = cpuSecond();

    // 2 - Compute threshold
    compute_threshold_gpu(gray_image, threshold_image, width, height, t_par);
    double time_threshold_gpu = cpuSecond() - t_start;
    printf("  Threshold done - time: %f\n", time_threshold_gpu);

    // 3 - Detect stars

    // 3.1 - Allocate memory for the stars info
    star *d_stars = nullptr;
    CHECK(cudaMallocManaged(&d_stars, max_stars * sizeof(star)));

    uint32_t *d_num_stars = nullptr;
    CHECK(cudaMallocManaged(&d_num_stars, sizeof(uint32_t)));
    *d_num_stars = 0;   // initialized to 0
    
    // 3.2 - Detect
    t_start = cpuSecond();
    detect_stars_gpu(threshold_image, width, height, max_star_size, min_star_size, d_stars, d_num_stars, max_stars);
    double time_detect_stars_gpu = cpuSecond() - t_start;
    printf("  Star detection done - time: %f\n", time_detect_stars_gpu);

    // 4 - Star details

    star_detail *stars_details = nullptr;
    CHECK(cudaMallocManaged(&stars_details, *d_num_stars * sizeof(star_detail)));

    t_start = cpuSecond();
    populate_star_details_gpu(stars_details, d_stars, *d_num_stars, fits_data, gray_image, width, npixels);
    double time_populate_star_details_gpu = cpuSecond() - t_start;
    printf("  Star details population done - time: %f\n", time_populate_star_details_gpu);

    // ==============================================
    // CPU part
    printf("\n======================================\n");
    printf("Running star detection on CPU\n\n");
    // allocating resources
    uint16_t *gray_image_cpu = new uint16_t[npixels];
    uint8_t *threshold_image_cpu = new uint8_t[npixels];

    // prefetch fits_data to CPU
    CHECK(cudaMemPrefetchAsync(fits_data, totpixels * sizeof(uint16_t), cudaCpuDeviceId, 0));
    CHECK(cudaDeviceSynchronize());

    // 1 - grayscale
    t_start = cpuSecond();
    to_grayscale_planar_cpu(fits_data, gray_image_cpu, npixels);
    double time_grayscale_cpu = cpuSecond() - t_start;
    printf("  Grayscale done - time: %f\n", time_grayscale_cpu);

    // 2 - threshold
    t_start = cpuSecond();
    compute_threshold_cpu(gray_image_cpu, threshold_image_cpu, width, height, t_par);
    double time_threshold_cpu = cpuSecond() - t_start;
    printf("  Threshold done - time: %f\n", time_threshold_cpu);
    
    // 3 - detection
    star *stars_cpu = new star[max_stars];
    uint32_t num_stars_cpu = 0;
    t_start = cpuSecond();
    detect_stars_cpu(threshold_image_cpu, width, height, max_star_size, min_star_size, stars_cpu, num_stars_cpu, max_stars);
    double time_detect_stars_cpu = cpuSecond() - t_start;
    printf("  Star detection done - time: %f\n", time_detect_stars_cpu);

    // 4 - details
    star_detail *stars_details_cpu = new star_detail[num_stars_cpu];
    t_start = cpuSecond();
    populate_star_details(stars_details_cpu, stars_cpu, num_stars_cpu, fits_data, gray_image_cpu, width, npixels);
    double time_populate_star_details_cpu = cpuSecond() - t_start;
    printf("  Star details population on CPU done - time: %f\n", time_populate_star_details_cpu);


    // ====================================================
    // Results comparaison
    printf("\n======================================\n");
    printf("Result comparaison\n\n");

    uint64_t matching_pixels = 0;
    uint64_t different_pixels = 0;
    
    for (uint64_t i = 0; i < npixels; i++) {
        if (threshold_image[i] == threshold_image_cpu[i]) {
            matching_pixels++;
        } else {
            different_pixels++;
        }
    }

    printf("  Matching pixels:  %lu (%.2f%%)\n", matching_pixels, 
           (double)matching_pixels / (double)npixels * 100.0);
    printf("  Different pixels: %lu (%.2f%%)\n\n", different_pixels,
           (double)different_pixels / (double)npixels * 100.0);

    printf("  Stars detected on GPU: %u, Stars detected on CPU: %u\n", *d_num_stars, num_stars_cpu);
    if (*d_num_stars != num_stars_cpu) {
        printf("  WARNING: Different number of stars detected!\n");
    }

    for (int i = 0; i < std::min((int)*d_num_stars, (int)num_stars_cpu); i++) {
        bool found = false;

        for (int j = 0; j < (int)num_stars_cpu; j++) {
            if (d_stars[i].start_x == stars_cpu[j].start_x && d_stars[i].start_y == stars_cpu[j].start_y &&
                d_stars[i].size_x == stars_cpu[j].size_x && d_stars[i].size_y == stars_cpu[j].size_y) {
                found = true;
                break;
            }
        }

        if (!found) {
            printf("  CPU detected star %u has no matches in GPU detected starts.\n", i);
        }
    }

    int matching_stars = 0;
    int matching_details = 0;
    for (int i = 0; i < std::min((int)*d_num_stars, (int)num_stars_cpu); i++) {
        bool found = false;
        bool matchesDetails = false;

        for (int j = 0; j < (int)*d_num_stars; j++) {
            if (d_stars[j].start_x == stars_cpu[i].start_x && d_stars[j].start_y == stars_cpu[i].start_y &&
                d_stars[j].size_x == stars_cpu[i].size_x && d_stars[j].size_y == stars_cpu[i].size_y) {
                
                found = true;
                matching_stars++;
                
                // Use epsilon for floating-point comparison
                const double eps = 1e-1;
                auto feq = [eps](double a, double b) { return fabs(a - b) < eps; };
                
                bool x_match = feq(stars_details[j].x, stars_details_cpu[i].x);
                bool y_match = feq(stars_details[j].y, stars_details_cpu[i].y);
                bool r_match = feq(stars_details[j].b_red, stars_details_cpu[i].b_red);
                bool g_match = feq(stars_details[j].b_green, stars_details_cpu[i].b_green);
                bool b_match = feq(stars_details[j].b_blue, stars_details_cpu[i].b_blue);
                bool total_match = feq(stars_details[j].b, stars_details_cpu[i].b);
                
                if (x_match && y_match && r_match && g_match && b_match && total_match) {
                    matching_details++;
                    matchesDetails = true;
                } else {
                    printf("  WARNING: GPU Star %u (CPU Star %u) details do not match!\n", i, j);
                    if (!x_match) printf("    x: GPU=%.6f CPU=%.6f diff=%.6f\n", stars_details[j].x, stars_details_cpu[i].x, fabs(stars_details[j].x - stars_details_cpu[i].x));
                    if (!y_match) printf("    y: GPU=%.6f CPU=%.6f diff=%.6f\n", stars_details[j].y, stars_details_cpu[i].y, fabs(stars_details[j].y - stars_details_cpu[i].y));
                    if (!r_match) printf("    b_red: GPU=%.6f CPU=%.6f diff=%.6f\n", stars_details[j].b_red, stars_details_cpu[i].b_red, fabs(stars_details[j].b_red - stars_details_cpu[i].b_red));
                    if (!g_match) printf("    b_green: GPU=%.6f CPU=%.6f diff=%.6f\n", stars_details[j].b_green, stars_details_cpu[i].b_green, fabs(stars_details[j].b_green - stars_details_cpu[i].b_green));
                    if (!b_match) printf("    b_blue: GPU=%.6f CPU=%.6f diff=%.6f\n", stars_details[j].b_blue, stars_details_cpu[i].b_blue, fabs(stars_details[j].b_blue - stars_details_cpu[i].b_blue));
                    if (!total_match) printf("    b: GPU=%.6f CPU=%.6f diff=%.6f\n", stars_details[j].b, stars_details_cpu[i].b, fabs(stars_details[j].b - stars_details_cpu[i].b));
                }

                break;
            }
        }

        if (!found) {
            printf("  GPU detected star %u has no matches in CPU detected starts.\n", i);
        }
    }
    printf("  Matching stars: %d\n", matching_stars);
    printf("  Matching stars details: %d\n", matching_details);
        
    // ====================================================
    // Performance comparaison
    printf("\n======================================\n");
    printf("Performance comparaison\n\n");

    printf("  GPU Grayscale time: %f s,\t CPU Grayscale time: %f s,\t speedup: %f x\n", 
        time_grayscale_gpu, time_grayscale_cpu, time_grayscale_cpu / time_grayscale_gpu);
    
    printf("  GPU Threshold time: %f s,\t CPU Threshold time: %f s,\t speedup: %f x\n", 
        time_threshold_gpu, time_threshold_cpu, time_threshold_cpu / time_threshold_gpu);
    
    printf("  GPU Star detection time: %f s,\t CPU Star detection time: %f s,\t speedup: %f x\n", 
        time_detect_stars_gpu, time_detect_stars_cpu, time_detect_stars_cpu / time_detect_stars_gpu);
    
    printf("  GPU Star details time: %f s,\t CPU Star details time: %f s,\t speedup: %f x\n", 
        time_populate_star_details_gpu, time_populate_star_details_cpu, time_populate_star_details_cpu / time_populate_star_details_gpu);
    
    double total_time_cpu = time_grayscale_cpu + time_threshold_cpu + time_detect_stars_cpu + time_populate_star_details_cpu;
    double total_time_gpu = time_grayscale_gpu + time_threshold_gpu + time_detect_stars_gpu + time_populate_star_details_gpu;
    printf("\n  Total GPU time: %f s,\t\t Total CPU time: %f s,\t\t speedup: %f x\n", 
        total_time_gpu, total_time_cpu, total_time_cpu / total_time_gpu);




    // write threshold image to disk
    const char *threshold_dir = "output_gray";

    cv::Mat img((int)height, (int)width, CV_8UC1, (void*)threshold_image);

    std::string out_path = std::string(threshold_dir) + "/threshold.png";
    if (!cv::imwrite(out_path, img)) {
        fprintf(stderr, "Failed to write PNG %s\n", out_path.c_str());
    } else {
        printf("\nSaved PNG %s\n", out_path.c_str());
    }


    // --- Print the stars info ---
    /*
    printf("Detected %u stars:\n", *d_num_stars);
    for (u_int32_t i = 0; i < std::min((int)*d_num_stars, 5); i++) {
        printf("Star %u: start=(%lu, %lu), size=(%u, %u)\n", i, d_stars[i].start_x, height - d_stars[i].start_y, d_stars[i].size_x, d_stars[i].size_y);
        printf("    Baricenter: (%.2f, %.2f), Brightness: R=%.2f G=%.2f B=%.2f Total=%.2f\n", 
               stars_details[i].x, height - stars_details[i].y, 
               stars_details[i].b_red, stars_details[i].b_green, stars_details[i].b_blue, stars_details[i].b);
    }
    if (*d_num_stars > 5)
        printf(".....\n");
    */

    // --- Draw stars on the image ---
    draw_stars(fits_data, width, d_stars, *d_num_stars);

    // --- Save images ---
    const char *detect_dir = "output_star";
    save_image_fits(detect_dir, "detect_output", fits_data, width, height, channels);
}
