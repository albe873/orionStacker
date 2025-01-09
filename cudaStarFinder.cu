#include "cuda_runtime.h"
#include "fits_api.h"
#include "device_thresholding.h"

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


int main(int argc, char **argv) {

    char *filename = nullptr;
    int opt, option_index = 0;
    u_int16_t threshold = 1000;
    u_int8_t reduce_factor = 8;
    u_int16_t window_size = 255;
    u_int16_t max_star_size = 75;

    enum ThresholdType {
        TR_SIMPLE,
        TR_ADAPTIVE,
        TR_ADAPTIVE_APPRISSIMATIVE
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
                threshold = atoi(optarg);
                break;
            case 'r':
                reduce_factor = atoi(optarg);
                break;
            case 'a':
                if (strcmp(optarg, "simple") == 0) {
                    threshold_algorithm = TR_SIMPLE;
                } else if (strcmp(optarg, "adaptive") == 0) {
                    threshold_algorithm = TR_ADAPTIVE;
                } else if (strcmp(optarg, "adaptive-approssimative") == 0) {
                    threshold_algorithm = TR_ADAPTIVE_APPRISSIMATIVE;
                } else {
                    fprintf(stderr, "Invalid threshold algorithm, using default\n");
                }
                break;
            case 'w':
                window_size = atoi(optarg);
                break;
            case 'm':
                max_star_size = atoi(optarg);
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
    u_int64_t totpixels = width * height * depth;
    u_int64_t npixels = width * height;

    u_int16_t *fits_data = nullptr;
    CHECK(cudaMallocManaged(&fits_data, totpixels * sizeof(u_int16_t)));
    CHECK(cudaMemPrefetchAsync(fits_data, totpixels * sizeof(u_int16_t), dev));

    u_int16_t *gray_image = nullptr;
    CHECK(cudaMallocManaged(&gray_image, npixels * sizeof(u_int16_t)));
    CHECK(cudaMemPrefetchAsync(gray_image, npixels * sizeof(u_int16_t), dev));

    u_int16_t *reduced_image = nullptr;
    CHECK(cudaMallocManaged(&reduced_image, (npixels / reduce_factor / reduce_factor) * sizeof(u_int16_t)));
    CHECK(cudaMemPrefetchAsync(reduced_image, (npixels / reduce_factor / reduce_factor) * sizeof(u_int16_t), dev));

    u_int16_t *threshold_image = nullptr;
    CHECK(cudaMallocManaged(&threshold_image, npixels * sizeof(u_int16_t)));
    CHECK(cudaMemPrefetchAsync(threshold_image, npixels * sizeof(u_int16_t), dev));

    //u_int16_t *star_map = nullptr;
    //CHECK(cudaMallocManaged(&star_map, npixels * sizeof(u_int16_t)));
    //CHECK(cudaMemPrefetchAsync(star_map, npixels * sizeof(u_int16_t), dev));

    get_fits_data(fptr, totpixels, fits_data);
    dim3 block_size_1d(256);
    dim3 grid_size_1d((npixels / 2 + block_size_1d.x - 1) / block_size_1d.x);

    dim3 block_size_2d(16, 16);
    dim3 grid_size_2d(  (width + block_size_2d.x - 1) / block_size_2d.x, 
                        (height + block_size_2d.y - 1) / block_size_2d.y
                    );

    // se depth == 1, allora bisogna applicare il filtro di bayer???
    to_grayscale_fits<<<grid_size_1d, block_size_1d>>>(fits_data, gray_image, npixels);
    CHECK(cudaDeviceSynchronize());

    switch (threshold_algorithm) {
        case TR_SIMPLE:
            simple_threshold<<<grid_size_1d, block_size_1d>>>(gray_image, threshold_image, npixels, threshold);
            break;
        case TR_ADAPTIVE:
            adaptiveThresholdingKernel<<<grid_size_2d, block_size_2d>>>(gray_image, threshold_image, width, height, window_size, threshold);
            break;
        case TR_ADAPTIVE_APPRISSIMATIVE:
            reduce_image<<<grid_size_2d, block_size_2d>>>(gray_image, reduced_image, width, height, reduce_factor);
            CHECK(cudaDeviceSynchronize());
            adaptiveThresholdingApprossimative<<<grid_size_2d, block_size_2d>>>(
                gray_image, threshold_image, width, height, reduced_image, reduce_factor, window_size, threshold);
            break;
    }
    CHECK(cudaDeviceSynchronize());

    save_image_fits("output_gray", threshold_image, width, height, 1);

    detect_stars<<<grid_size_2d, block_size_2d>>>(threshold_image, fits_data, width, height, max_star_size);
    CHECK(cudaDeviceSynchronize());

    save_image_fits("output_star", fits_data, width, height, 3);

}
