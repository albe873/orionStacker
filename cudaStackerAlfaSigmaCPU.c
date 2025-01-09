#include "fits_api.h"
#include "host_alfa_sigma.h"

#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <time.h>
#include <getopt.h>
#include <math.h>

/*
gcc cudaStackerAlfaSigmaCPU.c -o cudaStackerAlfaSigmaCPU -lcfitsio -lm -O3 -march=native -Wall
*/

double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

int main(int argc, char **argv) {

    // Parsing degli argomenti

    const char *in_dir = NULL;
    const char *out_dir = NULL;

    int opt, option_index = 0;
    float kappa = 3.0;
    int sigma = 5;

    static struct option long_options[] = {
        {"input-directory", required_argument, 0, 'i'},
        {"output-directory", required_argument, 0, 'o'},
        {"kappa", optional_argument, 0, 'k'},
        {"sigma", optional_argument, 0, 's'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "i:o:k:s", long_options, &option_index)) != -1) {
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

    if (in_dir == NULL || out_dir == NULL) {
        fprintf(stderr, "Usage: %s --input-directory </input/dir> --output-directory </output/dir>\n", argv[0]);
        return 1;
    }

    // Apertura della cartella
    remove_trailing_slash((char *)in_dir);
    DIR *dir;
    struct dirent *entry;
    if ((dir = opendir(in_dir)) == NULL) {
        perror("opendir");
        return 1;
    }

    // Scansione della cartella

    fitsfile *fptr = NULL;
    long width, height, depth, new_width, new_height, new_depth;
    int image_count = 0, image_num = 0, status;
    size_t npixels = 0;

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

    // Allocazione memoria

    u_int16_t **fits_data = NULL, *mean_CPU = NULL;
    u_int32_t *acc_CPU = NULL;
    float *std_CPU = NULL;

    fits_data = (u_int16_t **) malloc(image_num * sizeof(u_int16_t *));
    for (int i = 0; i < image_num; i++) {
        fits_data[i] = (u_int16_t *) malloc(npixels * sizeof(u_int16_t));
    }

    acc_CPU = (u_int32_t *) malloc(npixels * sizeof(u_int32_t));

    mean_CPU = (u_int16_t *) malloc(npixels * sizeof(u_int16_t));
    std_CPU = (float *) malloc(npixels * sizeof(float));

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

    // Calcola la media con algoritmo Alfa Sigma sulla CPU
    t_start = cpuSecond();
    for (int i = 0; i < sigma; i++) {
        computeMeanCPU(fits_data, mean_CPU, image_count, npixels);
        computeStdDevCPU(std_CPU, mean_CPU, fits_data, image_count, npixels);
        filterPixelsCPU(mean_CPU, std_CPU, fits_data, kappa, image_count, npixels);
    }
    computeMeanCPU(fits_data, mean_CPU, image_count, npixels);
    t_elapsed = cpuSecond() - t_start;
    printf("CPU Alfa Sigma elapsed time: %f\n", t_elapsed);
    save_image_fits(out_dir, mean_CPU, width, height, depth);


    // Pulizia della memoria
    free(mean_CPU);
    free(std_CPU);
    free(acc_CPU);
    for (int i = 0; i < image_count; i++) {
        free(fits_data[i]);
    }
    free(fits_data);

    exit(0);
}
