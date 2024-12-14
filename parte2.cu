#include "cuda_runtime.h"
#include "fits_api.h"

#include <stdio.h>
#include <getopt.h>
#include <unistd.h>

/*  --- AMD HIPify tool & HCC compiler ---

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


__global__ void simple_threshold(u_int16_t *image, int npixels, u_int16_t threshold) {
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    idx1 *= 2;
    int idx2 = idx1 + 1;

    if (idx2 < npixels) {
        u_int16_t val1 = image[idx1];
        u_int16_t val2 = image[idx2];

        image[idx1] = val1 > threshold ? 65535 : 0;
        image[idx2] = val2 > threshold ? 65535 : 0;
    }
}

int main(int argc, char **argv) {

    char *filename = nullptr;
    int opt, option_index = 0;

    static struct option long_options[] = {
        {"input-file", required_argument, 0, 'f'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "i:o:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'i':
                filename = optarg;
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

    get_fits_data(fptr, totpixels, fits_data);
    int block_size = 256;
    int grid_size = (npixels / 2 + block_size - 1) / block_size;

    // se depth == 1, allora bisogna applicare il filtro di bayer???

    to_grayscale_fits<<<grid_size, block_size>>>(fits_data, gray_image, npixels);
    CHECK(cudaDeviceSynchronize());
    save_image_fits("output_gray", gray_image, width, height, 1);
    
    sleep(2);

    simple_threshold<<<grid_size, block_size>>>(gray_image, npixels, 1500);
    CHECK(cudaDeviceSynchronize());

    save_image_fits("output_gray", gray_image, width, height, 1);
}
