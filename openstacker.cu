#include "cuda_runtime.h"
#include <stdio.h>
#include <fitsio.h>
#include <dirent.h>
#include <string.h>

// Include STB image libraries
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


/*  --- AMD HIPify tool & HCC compiler ---

--  to generate hipified code, use the following command
hipify-clang openstacker.cu --cuda-path=/opt/cuda

--  to compile the hipified code, use the following command
hipcc openstacker.cu.hip  -o openstacker -lcfitsio -O3 -Wall
*/

#define CHECK(err) do { cuda_check((err), __FILE__, __LINE__); } while(false)
inline void cuda_check(cudaError_t error_code, const char *file, int line) {
    if (error_code != cudaSuccess) {
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
        fflush(stderr);
        exit(error_code);
    }
}

// Funzione host per la conversione RGB->Gray
void rgbToGrayCPU(unsigned char *rgb, unsigned char *gray, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int rgbOffset = (y * width + x) * 3;
            int grayOffset = y * width + x;
            unsigned char r = rgb[rgbOffset];
            unsigned char g = rgb[rgbOffset + 1];
            unsigned char b = rgb[rgbOffset + 2];
            gray[grayOffset] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
        }
    }
}

__global__ void rgbToGrayGPU(unsigned char *d_rgb, unsigned char *d_gray, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        int grayIdx = y * width + x;
        d_gray[grayIdx] = (unsigned char)(0.299f * d_rgb[idx] + 0.587f * d_rgb[idx+1] + 0.114f * d_rgb[idx+2]);
    }
}

// accumulo dei valori di ogni pixel delle immagini con SATURAZIONE
__global__ void accumulatePixels(u_int32_t *acc_d, u_int16_t *d_image, int npixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < npixels) {
        acc_d[idx] += d_image[idx];
    }
}


// calcolo media finale
__global__ void computeMean(u_int32_t *acc_d, u_int16_t *mean, int numImages, int npixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < npixels) {
        mean[idx] = acc_d[idx] / numImages;
    }
}

void open_fits(char *file_path, fitsfile **fptr) {
    int status = 0;
    if (fits_open_file(fptr, file_path, READONLY, &status)) {
        fits_report_error(stderr, status);
        fprintf(stderr, "Could not open input file\n");
        exit(1);
    }
}

void get_image_dimensions(fitsfile *fptr, int *width, int *height, int *depth) {
    int status = 0;
    int naxis;
    long naxes[3] = {1, 1, 1};
    if (fits_get_img_dim(fptr, &naxis, &status) || fits_get_img_size(fptr, 3, naxes, &status)) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        exit(1);
    }
    *width = naxes[0];
    *height = naxes[1];
    if (naxis == 2) {
        *depth = 1;
        return;
    }
    *depth = naxes[2];
}


void get_fits_data(fitsfile *fptr, size_t npixels, u_int16_t *fits_data) {
    int status = 0;
    if (fits_read_img(fptr, TUSHORT, 1, npixels, NULL, fits_data, NULL, &status)) {
        fits_report_error(stderr, status);
        fprintf(stderr, "Could not read input file\n");
        fits_close_file(fptr, &status);
        free(fits_data);
        exit(1);
    }
}

void print_fits_metadata(fitsfile *fptr) {
    int nkeys, status = 0;
    char card[FLEN_CARD];
    if (fits_get_hdrspace(fptr, &nkeys, NULL, &status)) {
        fits_report_error(stderr, status);
        fprintf(stderr, "Could not get header space\n");
        exit(1);
    }

    printf("Header information:\n");
    for (int i = 1; i <= nkeys; i++) {
        if (fits_read_record(fptr, i, card, &status)) {
            fits_report_error(stderr, status);
            fprintf(stderr, "Could not read header record\n");
            exit(1);
        }
        printf("%s\n", card);
    }
}

void save_image_fits(char const *output_path, u_int16_t *image_data, int width, int height, int depth) {
    fitsfile *fptr;
    int status = 0;

    if (fits_create_file(&fptr, output_path, &status)) {
        if (status == FILE_NOT_CREATED) {
            status = 0; // Reset status before attempting to overwrite
            if (fits_open_file(&fptr, output_path, READWRITE, &status)) {
                fits_report_error(stderr, status);
                exit(1);
            }
            if (fits_delete_file(fptr, &status)) {
                fits_report_error(stderr, status);
                exit(1);
            }
            status = 0; // Reset status before creating the file again
            if (fits_create_file(&fptr, output_path, &status)) {
                fits_report_error(stderr, status);
                exit(1);
            }
        } else {
            fits_report_error(stderr, status);
            exit(1);
        }
    }

    if (depth == 1) {
        if (fits_create_img(fptr, USHORT_IMG, 2, (long[]){width, height}, &status)) {
            fits_report_error(stderr, status);
            exit(1);
        }
    } else {
        if (fits_create_img(fptr, USHORT_IMG, 3, (long[]){width, height, depth}, &status)) {
            fits_report_error(stderr, status);
            exit(1);
        }
    }

    // Write image data
    if (fits_write_img(fptr, TUSHORT, 1, width * height * depth, image_data, &status)) {
        fits_report_error(stderr, status);
        exit(1);
    }

    printf("Image saved to %s, metadata:\n", output_path);
    print_fits_metadata(fptr);
    fits_close_file(fptr, &status);
}

void convert_to_rgb(float *fits_data, size_t npixels, uint8_t *rgb) {
    for (size_t i = 0; i < npixels; i++) {
        uint8_t value = (uint8_t)(fits_data[i] / (65535.0 / 255.0));
        rgb[i * 3] = value;
        rgb[i * 3 + 1] = value;
        rgb[i * 3 + 2] = value;
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <directory_path>\n", argv[0]);
        return 1;
    }

    // Imposta il device CUDA
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev)); // Ottiene le proprietà del dispositivo CUDA
    CHECK(cudaSetDevice(0)); // Seleziona il dispositivo CUDA

    DIR *dir;
    struct dirent *entry;
    if ((dir = opendir(argv[1])) == NULL) {
        perror("opendir");
        return 1;
    }

    // Scansione della cartella

    fitsfile *fptr = nullptr;
    int width, height, depth, new_width, new_height, new_depth, image_count = 0, status, block_size = 256, grid_size;
    size_t npixels;
    u_int16_t *fits_data_h = nullptr, *fits_data_d = nullptr;
    u_int32_t *acc_d = nullptr;

    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG) {  // Controlla se è un file regolare
            // Costruisce il percorso completo
            char file_path[1024];
            snprintf(file_path, sizeof(file_path), "%s/%s", argv[1], entry->d_name);

            // Processa solo i file .fits
            if (strstr(file_path, ".fits") != NULL || strstr(file_path, ".fit")) {
                printf("Processing file: %s\n", file_path);
                open_fits(file_path, &fptr);

                if (image_count == 0) {
                    print_fits_metadata(fptr);
                    get_image_dimensions(fptr, &width, &height, &depth);
                    npixels = width * height * depth;
                    grid_size = (npixels + block_size - 1) / block_size;
                    CHECK(cudaMalloc(&fits_data_d, npixels * sizeof(u_int16_t)));
                    CHECK(cudaMalloc(&acc_d, npixels * sizeof(u_int32_t)));
                    CHECK(cudaMemset(acc_d, 0, npixels * sizeof(u_int32_t)));

                    fits_data_h = (u_int16_t *) malloc(npixels * sizeof(u_int16_t));
                    get_fits_data(fptr, npixels, fits_data_h);
                    fits_close_file(fptr, &status);
                }
                else {
                    get_image_dimensions(fptr, &new_width, &new_height, &new_depth);
                    if (new_width != width || new_height != height) {
                        fprintf(stderr, "Skipping file %s due to mismatched dimensions.\n", file_path);
                        fits_close_file(fptr, &status);
                        continue;
                    }

                    get_fits_data(fptr, npixels, fits_data_h);
                    fits_close_file(fptr, &status);

                    CHECK(cudaDeviceSynchronize()); // lazy cuda device synchronization
                }
                
                CHECK(cudaMemcpy(fits_data_d, fits_data_h, npixels * sizeof(u_int16_t), cudaMemcpyHostToDevice));
                accumulatePixels<<<grid_size, block_size>>>(acc_d, fits_data_d, npixels);

                image_count++;
            }
        }
    }
    closedir(dir);

    // Calcola la media finale
    CHECK(cudaDeviceSynchronize()); // lazy cuda device synchronization
    computeMean<<<grid_size, block_size>>>(acc_d, fits_data_d, image_count, npixels);

    // Copia il risultato finale sull'host
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(fits_data_h, fits_data_d, npixels * sizeof(u_int16_t), cudaMemcpyDeviceToHost));

    // Converte i dati FITS in RGB
    //uint8_t *rgb = (uint8_t *) malloc(npixels * 3 * sizeof(uint8_t));
    //convert_to_rgb(fits_data_h, npixels, rgb);

    //stbi_write_png("output/oputput_rgb.png", width, height, 3, rgb, width * 3);

    save_image_fits("output/output.fits", fits_data_h, width, height, depth);



    CHECK(cudaDeviceReset());
    exit(0);
}
