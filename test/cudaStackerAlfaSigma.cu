#include "cuda_runtime.h"

#include "fits_api.h"
#include "cuda_check.h"

#include "device_alfa_sigma.h"
#include "host_alfa_sigma.h"

#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <ctime>
#include <getopt.h>
#include <cstdio>


/*  --- AMD HIPify tool & HCC compiler for AMD ROCM ---
--  to generate hipified code, use the following command
hipify-clang cudaStackerAlfaSigma.cu --cuda-path=/opt/cuda

--  to compile the hipified code, use the following command
hipcc cudaStackerAlfaSigma.cu.hip  -o cudaStackerAlfaSigma -lcfitsio -O3 -Wall
*/

/*  --- NVCC compiler for NVIDIA CUDA -
--  to compile for cuda, use the following command
nvcc cudaStackerAlfaSigma.cu -o cudaStackerAlfaSigma -lcfitsio -O3
*/

double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

int main(int argc, char **argv) {

    // Parsing degli argomenti

    const char *in_dir = NULL;
    const char *out_dir = ".";
    const char *file_name = "image";

    int opt, option_index = 0;
    long num;
    float numf;
    char *end;

    float kappa = 3.0;
    u_int16_t sigma = 5;

    static struct option long_options[] = {
        {"input-directory", required_argument, 0, 'i'},
        {"output-directory", optional_argument, 0, 'o'},
        {"file-name", optional_argument, 0, 'n'},
        {"kappa", optional_argument, 0, 'k'},
        {"sigma", optional_argument, 0, 's'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "i:o:n:k:s", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'i':
                in_dir = optarg;
                break;
            case 'o':
                out_dir = optarg;
                break;
            case 'n':
                file_name = optarg;
                break;
            case 'k':
                numf = strtof(optarg, &end);
                if (end == optarg) {
                    fprintf(stderr, "Invalid argument for kappa, using default\n");
                } else if (numf < 0 || numf > 100) {
                    fprintf(stderr, "Invalid argument for kappa, using default\n");
                } else {
                    kappa = numf;
                }
                break;
            case 's':
                num = strtol(optarg, &end, 10);
                if (end == optarg) {
                    fprintf(stderr, "Invalid argument for sigma, using default\n");
                } else if (num < 0 || num > 65535) {
                    fprintf(stderr, "Invalid argument for sigma, using default\n");
                } else {
                    sigma = num;
                }
                break;
            default:
                fprintf(stderr, "Usage: %s --input-directory <input/dir>\n", argv[0]);
                return 1;
        }
    }

    if (in_dir == nullptr || out_dir == nullptr) {
        fprintf(stderr, "Usage: %s --input-directory </input/dir>\n", argv[0]);
        return 1;
    }


    // Imposta il device CUDA
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev)); // Ottiene le proprietà del dispositivo CUDA
    CHECK(cudaSetDevice(0)); // Seleziona il dispositivo CUDA
    cudaMemLocation devLoc;
    devLoc.type = cudaMemLocationTypeDevice;

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
    long width, height, n_chan, new_width, new_height, new_n_chan;
    int status;
    u_int16_t image_count = 0, image_num = 0;
    dim3 block_size(512), grid_size;
    u_int64_t npixels;

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
                    get_image_dimensions(fptr, &width, &height, &n_chan);
                    npixels = width * height * n_chan;
                    grid_size = (npixels / 2 + block_size.x - 1) / block_size.x;
                    printf("grid_size: %d, tot %d, wasted %lu\n", grid_size.x, grid_size.x * block_size.x, (grid_size.x * block_size.x) - npixels/2);
                }
                else {
                    get_image_dimensions(fptr, &new_width, &new_height, &new_n_chan);
                    if (new_width != width || new_height != height || new_n_chan != n_chan) {
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

    // Controllo del numero di immagini
    if (image_num == 0) {
        fprintf(stderr, "No valid images found in directory %s\n", in_dir);
        exit(1);
    } else if (image_num == 1) {
        fprintf(stderr, "Only one image found, no stacking needed\n");
        exit(0);
    } else {
        printf("Found %d images\n", image_num);
    }

    // Allocazione memoria unificata

    u_int16_t **fits_data = nullptr, *mean = nullptr;

    CHECK(cudaMallocManaged(&fits_data, image_num * sizeof(u_int16_t*)));
    for (int i = 0; i < image_num; i++) {
        CHECK(cudaMallocManaged(&fits_data[i], npixels * sizeof(u_int16_t)));
    }

    CHECK(cudaMallocManaged(&mean, npixels * sizeof(u_int16_t)));
    CHECK(cudaMemAdvise(mean, npixels * sizeof(u_int16_t), cudaMemAdviseSetPreferredLocation, devLoc));

    // Lettura dei file .fits e caricamento dei dati in memoria (unificata)

    if ((dir = opendir(in_dir)) == NULL) {
        perror("opendir");
        return 1;
    }
    while ((entry = readdir(dir)) != NULL && image_count < image_num) {
        if (entry->d_type == DT_REG) {  // Controlla se è un file regolare
            // Costruisce il percorso completo
            char file_path[1024];
            snprintf(file_path, sizeof(file_path), "%s/%s", in_dir, entry->d_name);

            // Processa solo i file .fits
            if (strstr(file_path, ".fits") != NULL || strstr(file_path, ".fit")) {
                printf("Opening file: %s\n", file_path);
                open_fits(file_path, &fptr);

                get_image_dimensions(fptr, &new_width, &new_height, &new_n_chan);
                if (new_width != width || new_height != height || new_n_chan != n_chan) {
                    fprintf(stderr, "Skipping file %s due to mismatched dimensions.\n", file_path);
                    fits_close_file(fptr, &status);
                    continue;
                }

                get_fits_data(fptr, npixels, fits_data[image_count]);
                CHECK(cudaMemPrefetchAsync(fits_data[image_count], npixels * sizeof(u_int16_t), devLoc, 0));
                fits_close_file(fptr, &status);
                image_count++;
            }
        }
    }
    
    closedir(dir);
    double t_start, t_elapsed;

    // Calcola la media con algoritmo Alfa Sigma
    
    printf("Computing mean with Alfa Sigma with GPU ...\n");
    t_start = cpuSecond();

    compute_alfa_sigma_uint16<<<grid_size, block_size>>>(fits_data, mean, image_count, npixels, kappa, sigma);
    CHECK(cudaGetLastError());  // Check for kernel launch errors

    CHECK(cudaDeviceSynchronize());
    
    t_elapsed = cpuSecond() - t_start;
    printf("GPU Alfa Sigma elapsed time: %f\n", t_elapsed);
    
    
    save_image_fits(out_dir, file_name, mean, width, height, n_chan);


    // free memory
    for (int i = 0; i < image_num; i++) {
        CHECK(cudaFree(fits_data[i]));
    }
    CHECK(cudaFree(fits_data));
    CHECK(cudaFree(mean));

    CHECK(cudaDeviceReset());

    exit(0);
}
