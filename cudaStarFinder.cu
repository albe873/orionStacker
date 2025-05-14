#include "cuda_runtime.h"
#include "fits_api.h"
#include "device_starFinder.h"

#include <stdio.h>
#include <getopt.h>
#include <unistd.h>

/*  --- AMD HIPify tool & HIPCC compiler ---

--  to generate hipified code, use the following command
hipify-clang cudaStarFinder.cu --cuda-path=/opt/cuda

--  to compile the hipified code, use the following command
hipcc cudaStarFinder.cu.hip  -o cudaStarFinder -lcfitsio -O3 -Wall
*/

/*  --- NVCC compiler ---

--  to compile for cuda, use the following command
nvcc cudaStarFinder.cu -o cudaStarFinder -lcfitsio -O3
*/


#define CHECK(err) do { cuda_check((err), __FILE__, __LINE__); } while(false)
inline void cuda_check(cudaError_t error_code, const char *file, int line) {
    if (error_code != cudaSuccess) {
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
        fflush(stderr);
        exit(error_code);
    }
}

double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}


int main(int argc, char **argv) {

    char *filename = nullptr;
    int opt, option_index = 0;
    long num;
    char *end;
    u_int16_t threshold = 1000;
    u_int16_t reduce_factor = 8;
    u_int16_t window_size = 255;
    u_int16_t max_star_size = 75;

    enum ThresholdType {
        TR_SIMPLE,
        TR_ADAPTIVE,
        TR_FAST_ADAPTIVE
    };
    ThresholdType threshold_algorithm = TR_SIMPLE;

    static struct option long_options[] = {
        {"input-file", required_argument, 0, 'f'},
        {"threshold", optional_argument, 0, 't'},
        {"reduce-factor", optional_argument, 0, 'r'},
        {"threshold-algorith", optional_argument, 0, 'a'},
        {"window-size", optional_argument, 0, 'w'},
        {"max-star-size", optional_argument, 0, 'm'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "f:t:r:a:w:m:", long_options, &option_index)) != -1) {
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
                    threshold = num;
                }
                break;
            case 'r':
                num = strtol(optarg, &end, 10);
                if (end == optarg) {
                    fprintf(stderr, "Cannot convert reduce factor, using default\n");
                } else if (num < 1 || num > 65535) {
                    fprintf(stderr, "Invalid reduce factor, using default\n");
                } else {
                    reduce_factor = num;
                }
                break;
            case 'a':
                if (strcmp(optarg, "simple") == 0) {
                    threshold_algorithm = TR_SIMPLE;
                } else if (strcmp(optarg, "adaptive") == 0) {
                    threshold_algorithm = TR_ADAPTIVE;
                } else if (strcmp(optarg, "fast-adaptive") == 0) {
                    threshold_algorithm = TR_FAST_ADAPTIVE;
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
                    window_size = num;
                }
                break;
            case 'm':
                num = strtol(optarg, &end, 10);
                if (end == optarg) {
                    fprintf(stderr, "Cannot convert max star size, using default\n");
                } else if (num < 1 || num > 65535) {
                    fprintf(stderr, "Invalid max star size, using default\n");
                } else {
                    max_star_size = num;
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

    // Imposta il device CUDA

    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev)); // Ottiene le proprietà del dispositivo CUDA
    CHECK(cudaSetDevice(0)); // Seleziona il dispositivo CUDA

    // Apre il file FITS
    fitsfile *fptr = nullptr;
    long width, height, depth;
    open_fits(filename, &fptr);
    get_image_dimensions(fptr, &width, &height, &depth);

    if (width * height * depth == 0) {
        fprintf(stderr, "Invalid image dimensions\n");
        return 1;
    }
    u_int64_t totpixels = width * height * depth;
    u_int64_t npixels = width * height;

    u_int16_t *fits_data = nullptr;
    CHECK(cudaMallocManaged(&fits_data, totpixels * sizeof(u_int16_t)));

    u_int16_t *gray_image = nullptr;
    CHECK(cudaMallocManaged(&gray_image, npixels * sizeof(u_int16_t)));
    CHECK(cudaMemPrefetchAsync(gray_image, npixels * sizeof(u_int16_t), dev));

    u_int16_t *reduced_image = nullptr;
    if (threshold_algorithm == TR_FAST_ADAPTIVE) {
        CHECK(cudaMallocManaged(&reduced_image, (npixels / reduce_factor / reduce_factor) * sizeof(u_int16_t)));
        CHECK(cudaMemPrefetchAsync(reduced_image, (npixels / reduce_factor / reduce_factor) * sizeof(u_int16_t), dev));
    }

    u_int16_t *threshold_image = nullptr;
    CHECK(cudaMallocManaged(&threshold_image, npixels * sizeof(u_int16_t)));
    CHECK(cudaMemPrefetchAsync(threshold_image, npixels * sizeof(u_int16_t), dev));

    get_fits_data(fptr, totpixels, fits_data);
    CHECK(cudaMemPrefetchAsync(fits_data, totpixels * sizeof(u_int16_t), dev));
    
    // --- grid and block sizes ---
    dim3 block_size_1d(256);
    dim3 grid_size_1d((npixels / 2 + block_size_1d.x - 1) / block_size_1d.x);

    dim3 block_size_2d(16, 16);
    dim3 grid_size_2d(  (width + block_size_2d.x - 1) / block_size_2d.x, 
                        (height + block_size_2d.y - 1) / block_size_2d.y
                    );

    double t_start, t_elapsed;
    t_start = cpuSecond();

    // --- Convert to grayscale ---
    if (depth == 3) {
        to_grayscale_fits<<<grid_size_1d, block_size_1d>>>(fits_data, gray_image, npixels);
        CHECK(cudaDeviceSynchronize());
    }
    else // depth == 1
        CHECK(cudaMemcpy(gray_image, fits_data, npixels * sizeof(u_int16_t), cudaMemcpyDeviceToDevice));

    // --- Apply thresholding ---
    switch (threshold_algorithm) {
        case TR_SIMPLE:
            simple_threshold<<<grid_size_1d, block_size_1d>>>(gray_image, threshold_image, npixels, threshold);
            break;
        case TR_ADAPTIVE:
            adaptiveThresholdingKernel<<<grid_size_2d, block_size_2d>>>(gray_image, threshold_image, width, height, window_size, threshold);
            break;
        case TR_FAST_ADAPTIVE:
            reduce_image<<<grid_size_2d, block_size_2d>>>(gray_image, reduced_image, width, height, reduce_factor);
            CHECK(cudaDeviceSynchronize());
            adaptiveThresholdingApprossimative<<<grid_size_2d, block_size_2d>>>(
                gray_image, threshold_image, width, height, reduced_image, reduce_factor, window_size, threshold);
            break;
    }
    CHECK(cudaDeviceSynchronize());

    // --- Detect stars ---

    //detect_stars<<<grid_size_2d, block_size_2d>>>(threshold_image, fits_data, width, height, max_star_size);
    new_detect_stars<<<grid_size_2d, block_size_2d>>>(threshold_image, fits_data, width, height, max_star_size);
    CHECK(cudaDeviceSynchronize());

    t_elapsed = cpuSecond() - t_start;
    printf("Elapsed time: %f\n", t_elapsed);

    // --- Save images ---
    const char *threshold_dir = "output_gray";
    const char *detect_dir = "output_star";

    save_image_fits(threshold_dir, "threshold", threshold_image, width, height, 1);
    save_image_fits(detect_dir, "detect_output", fits_data, width, height, depth);
}
