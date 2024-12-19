#include "cuda_runtime.h"
#include "fits_api.h"

#include <stdio.h>
#include <getopt.h>
#include <unistd.h>

/*  --- AMD HIPify tool & HIPCC compiler ---

--  to generate hipified code, use the following command
hipify-clang parte2.cu --cuda-path=/opt/cuda

--  to compile the hipified code, use the following command
hipcc parte2.cu.hip  -o parte2 -lcfitsio -O3 -Wall
*/

/*
--  to compile for cuda, use the following command
nvcc parte2.cu -o parte2 -lcfitsio -O3
*/



#define CHECK(err) do { cuda_check((err), __FILE__, __LINE__); } while(false)
inline void cuda_check(cudaError_t error_code, const char *file, int line) {
    if (error_code != cudaSuccess) {
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
        fflush(stderr);
        exit(error_code);
    }
}

// fits data is in planar format
__global__ void to_grayscale_fits(u_int16_t *image, u_int16_t *gray_image, int npixels) {
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    idx1 *= 2;
    int idx2 = idx1 + 1;

    if (idx2 < npixels) {
        u_int16_t red1 = image[idx1];
        u_int16_t red2 = image[idx2];

        u_int16_t green1 = image[idx1 + npixels];
        u_int16_t green2 = image[idx2 + npixels];

        u_int16_t blue1 = image[idx1 + 2*npixels];
        u_int16_t blue2 = image[idx2 + 2*npixels];

        gray_image[idx1] = 0.299*red1 + 0.587*green1 + 0.114*blue1;
        gray_image[idx2] = 0.299*red2 + 0.587*green2 + 0.114*blue2;
    }
}


__global__ void simple_threshold(u_int16_t *image, u_int16_t *output, int npixels, u_int16_t threshold) {
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    idx1 *= 2;
    int idx2 = idx1 + 1;

    if (idx2 < npixels) {
        u_int16_t val1 = image[idx1];
        u_int16_t val2 = image[idx2];

        output[idx1] = val1 > threshold ? 65535 : 0;
        output[idx2] = val2 > threshold ? 65535 : 0;
    }
}

__global__ void adaptiveThresholdingKernel(u_int16_t *image, u_int16_t *output, int width, int height, u_int16_t windowSize, u_int16_t offset) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        windowSize /= 2;
        int startX = max(x - windowSize, 0);
        int endX = min(x + windowSize, width);
        int startY = max(y - windowSize, 0);
        int endY = min(y + windowSize, height);

        // Calcola la media del blocco locale
        u_int32_t sum = 0;
        for (u_int32_t i = startY; i < endY; i++) {
            for (u_int32_t j = startX; j < endX; j++) {
                sum += image[i * width + j];
            }
        }
        int localMean = sum / ((endX - startX) * (endY - startY));
        int pixel = image[y * width + x];

        // Applica il thresholding adattivo
        output[y * width + x] = (pixel > (localMean + offset)) ? 65535 : 0;
    }
}

__global__ void reduce_image(u_int16_t *image, u_int16_t *reduced_image, int width, int height, int reduce_factor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int new_width = width / reduce_factor;
    int new_height = height / reduce_factor;

    if (x < new_width && y < new_height) {
        u_int32_t sum = 0;
        for (int i = 0; i < reduce_factor; i++) {
            for (int j = 0; j < reduce_factor; j++) {
                int orig_x = x * reduce_factor + i;
                int orig_y = y * reduce_factor + j;
                if (orig_x >= width || orig_y >= height) {
                    continue;
                }
                sum += image[orig_y * width + orig_x];
            }
        }
        reduced_image[y * new_width + x] = sum / (reduce_factor * reduce_factor);
    }
}


int main(int argc, char **argv) {

    char *filename = nullptr;
    int opt, option_index = 0;
    int threshold = 1000;
    int reduce_factor = 8;
    int window_size = 255;

    enum ThresholdType {
        TR_SIMPLE,
        TR_ADAPTIVE
    };
    ThresholdType threshold_algorithm = TR_SIMPLE;

    static struct option long_options[] = {
        {"input-file", required_argument, 0, 'f'},
        {"threshold", optional_argument, 0, 't'},
        {"reduce-factor", optional_argument, 0, 'r'},
        {"threshold-algorith", optional_argument, 0, 'a'},
        {"window-size", optional_argument, 0, 'w'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "f:t:r:a:w:", long_options, &option_index)) != -1) {
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
                } else {
                    fprintf(stderr, "Invalid threshold algorithm, using default\n");
                }
                break;
            case 'w':
                window_size = atoi(optarg);
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
    int width, height, depth;
    open_fits(filename, &fptr);
    get_image_dimensions(fptr, &width, &height, &depth);
    int totpixels = width * height * depth;
    int npixels = width * height;

    u_int16_t *fits_data = nullptr;
    CHECK(cudaMallocManaged(&fits_data, totpixels * sizeof(u_int16_t)));
    CHECK(cudaMemPrefetchAsync(fits_data, totpixels * sizeof(u_int16_t), dev));

    u_int16_t *gray_image = nullptr;
    CHECK(cudaMallocManaged(&gray_image, npixels * sizeof(u_int16_t)));
    CHECK(cudaMemPrefetchAsync(gray_image, npixels * sizeof(u_int16_t), dev));

    u_int16_t *reduced_image = nullptr;
    CHECK(cudaMallocManaged(&reduced_image, (npixels / reduce_factor / reduce_factor) * sizeof(u_int16_t)));
    CHECK(cudaMemPrefetchAsync(reduced_image, (npixels / reduce_factor / reduce_factor) * sizeof(u_int16_t), dev));

    u_int16_t *output_image = nullptr;
    CHECK(cudaMallocManaged(&output_image, npixels * sizeof(u_int16_t)));
    CHECK(cudaMemPrefetchAsync(output_image, npixels * sizeof(u_int16_t), dev));

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

    reduce_image<<<grid_size_2d, block_size_2d>>>(gray_image, reduced_image, width, height, reduce_factor);
    CHECK(cudaDeviceSynchronize());
    //save_image_fits("output_gray", reduced_image, width/reduce_factor, height/reduce_factor, 1);

    sleep(2);
    switch (threshold_algorithm) {
        case TR_SIMPLE:
            simple_threshold<<<grid_size_1d, block_size_1d>>>(gray_image, output_image, npixels, threshold);
            break;
        case TR_ADAPTIVE:
            adaptiveThresholdingKernel<<<grid_size_2d, block_size_2d>>>(gray_image, output_image, width, height, window_size, threshold);
            break;
    }
    CHECK(cudaDeviceSynchronize());

    save_image_fits("output_gray", output_image, width, height, 1);
}
