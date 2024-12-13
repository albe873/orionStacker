#include "cuda_runtime.h"
#include "fits_api.h"
#include "device_alfa_sigma.h"
#include "host_alfa_sigma.h"

#include <stdio.h>
#include <fitsio.h>
#include <dirent.h>
#include <string.h>
#include <ctime>
#include <getopt.h>
#include <cstdio>


/*  --- AMD HIPify tool & HCC compiler ---

--  to generate hipified code, use the following command
hipify-clang cudaStackerAlfaSigma.cu --cuda-path=/opt/cuda

--  to compile the hipified code, use the following command
hipcc cudaStackerAlfaSigma.cu.hip  -o cudaStackerAlfaSigma -lcfitsio -O3 -Wall
*/

/*
--  to compile for cuda, use the following command
nvcc cudaStackerAlfaSigma.cu -o ucdaStackerAlfaSigma -lcfitsio -O3
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

    // Parsing degli argomenti

    const char *in_dir = nullptr;
    const char *out_dir = nullptr;

    int opt, option_index = 0;
    float kappa = 3.0, sigma = 2.0;

    static struct option long_options[] = {
        {"input-directory", required_argument, 0, 'i'},
        {"output-directory", required_argument, 0, 'o'},
        {"kappa", optional_argument, 0, 'k'},
        {"sigma", optional_argument, 0, 's'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "i:o:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'i':
                in_dir = optarg;
                break;
            case 'o':
                out_dir = optarg;
                break;
            case 'k':
                kappa = atoi(optarg);
                break;
            case 's': 
                sigma = atof(optarg);
                break;
            default:
                fprintf(stderr, "Usage: %s --input-directory <input/dir> --output-directory </output/dir>\n", argv[0]);
                return 1;
        }
    }

    if (in_dir == nullptr || out_dir == nullptr) {
        fprintf(stderr, "Usage: %s --input-directory </input/dir> --output-directory </output/dir>\n", argv[0]);
        return 1;
    }


    // Imposta il device CUDA

    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev)); // Ottiene le proprietà del dispositivo CUDA
    CHECK(cudaSetDevice(0)); // Seleziona il dispositivo CUDA

    // Apertura della cartella
    remove_trailing_slash((char *)in_dir);
    DIR *dir;
    struct dirent *entry;
    if ((dir = opendir(in_dir)) == NULL) {
        perror("opendir");
        return 1;
    }

    // Scansione della cartella

    fitsfile *fptr = nullptr;
    int width, height, depth, new_width, new_height, new_depth, image_count = 0, image_num = 0, status, block_size = 512, grid_size;
    size_t npixels;

    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG) {  // Controlla se è un file regolare
            // Costruisce il percorso completo
            char file_path[1024];
            snprintf(file_path, sizeof(file_path), "%s/%s", in_dir, entry->d_name);

            // Processa solo i file .fits
            if (strstr(file_path, ".fits") != NULL || strstr(file_path, ".fit")) {
                open_fits(file_path, &fptr);

                if (image_num == 0) {
                    print_fits_metadata(fptr);
                    get_image_dimensions(fptr, &width, &height, &depth);
                    npixels = width * height * depth;
                    grid_size = (npixels / 2 + block_size) / block_size;

                }
                else {
                    get_image_dimensions(fptr, &new_width, &new_height, &new_depth);
                    if (new_width != width || new_height != height || new_depth != depth) {
                        fprintf(stderr, "Skipping file %s due to mismatched dimensions.\n", file_path);
                        fits_close_file(fptr, &status);
                        continue;
                    }
                }

                fits_close_file(fptr, &status);
                image_num++;
            }
        }
    }
    closedir(dir);

    // Allocazione memoria unificata

    u_int16_t **fits_data = nullptr, *mean = nullptr;
    //u_int16_t *fits_data_CPU = nullptr;
    //u_int32_t *acc_CPU = nullptr;

    CHECK(cudaMallocManaged(&fits_data, image_num * sizeof(u_int16_t*)));
    for (int i = 0; i < image_num; i++) {
        CHECK(cudaMallocManaged(&fits_data[i], npixels * sizeof(u_int16_t)));
        CHECK(cudaMemPrefetchAsync(fits_data[i], npixels * sizeof(u_int16_t), dev));
    }

    CHECK(cudaMallocManaged(&mean, npixels * sizeof(u_int16_t)));
    CHECK(cudaMemAdvise(mean, npixels * sizeof(u_int16_t), cudaMemAdviseSetPreferredLocation, dev));

    // Lettura dei file .fits e caricamento dei dati in memoria (unificata)

    if ((dir = opendir(in_dir)) == NULL) {
        perror("opendir");
        return 1;
    }
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG) {  // Controlla se è un file regolare
            // Costruisce il percorso completo
            char file_path[1024];
            snprintf(file_path, sizeof(file_path), "%s/%s", in_dir, entry->d_name);

            // Processa solo i file .fits
            if (strstr(file_path, ".fits") != NULL || strstr(file_path, ".fit")) {
                printf("Opening file: %s\n", file_path);
                open_fits(file_path, &fptr);

                get_image_dimensions(fptr, &new_width, &new_height, &new_depth);
                if (new_width != width || new_height != height || new_depth != depth) {
                    fprintf(stderr, "Skipping file %s due to mismatched dimensions.\n", file_path);
                    fits_close_file(fptr, &status);
                    continue;
                }

                get_fits_data(fptr, npixels, fits_data[image_count]);
                fits_close_file(fptr, &status);
                image_count++;

            }
        }
    }
    closedir(dir);

    double t_start, t_elapsed;

    // Calcola la media con algoritmo Alfa Sigma
    ///*
    printf("Computing mean with Alfa Sigma with GPU ...\n");
    t_start = cpuSecond();
    /*
    for (int i = 0; i < 5; i++) {
        computeMeanAdv<<<grid_size, block_size>>>(fits_data, mean, image_count, npixels);
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemset(std, 0, npixels * sizeof(float)));
        computeStdDev<<<grid_size, block_size>>>(std, mean, fits_data, image_count, npixels);
        CHECK(cudaDeviceSynchronize());
        
        filterPixels<<<grid_size, block_size>>>(mean, std, fits_data, 3, image_count, npixels);
        CHECK(cudaDeviceSynchronize());
    }

    computeMeanAdv<<<grid_size, block_size>>>(fits_data, mean, image_count, npixels);
    */

    compute_alfa_sigma2<<<grid_size, block_size>>>(fits_data, mean, image_count, npixels, kappa, sigma);   
    CHECK(cudaDeviceSynchronize());
    t_elapsed = cpuSecond() - t_start;
    printf("GPU Alfa Sigma elapsed time: %f\n", t_elapsed);
    
    save_image_fits(out_dir, mean, width, height, depth);
    //*/

    // Calcola la media con algoritmo Alfa Sigma sulla CPU
    /*
    u_int16_t *meanCPU = (u_int16_t *) malloc(npixels * sizeof(u_int16_t));
    float *stdCPU = (float *) malloc(npixels * sizeof(float));
    t_start = cpuSecond();
    for (int i = 0; i < 5; i++) {
        computeMeanAdvCPU(fits_data, meanCPU, image_count, npixels);
        computeStdDevCPU(stdCPU, meanCPU, fits_data, image_count, npixels);
        filterPixelsCPU(meanCPU, stdCPU, fits_data, 3, image_count, npixels);
    }
    computeMeanAdvCPU(fits_data, meanCPU, image_count, npixels);
    t_elapsed = cpuSecond() - t_start;
    printf("CPU Alfa Sigma elapsed time: %f\n", t_elapsed);
    save_image_fits("output/image", meanCPU, width, height, depth);
    free(meanCPU);
    free(stdCPU);
    //*/

    // Calcola la media con accumulo dei pixel
    for (int i = 0; i < image_num; i++) {
        CHECK(cudaFree(fits_data[i]));
    }
    CHECK(cudaFree(fits_data));
    CHECK(cudaFree(mean));

    CHECK(cudaDeviceReset());
    exit(0);
}
