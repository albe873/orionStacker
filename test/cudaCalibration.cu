#include "cuda_runtime.h"
#include "fits_api.h"
#include "cuda_check.h"
#include "calibration.h"

#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <ctime>
#include <getopt.h>

/* 

eseguire con:
./cudaCalibration --light /home/user/Downloads/dss_3chan/input/ --bias /home/user/Downloads/dss_3chan/bias --dark /home/user/Downloads/dss_3chan/dark/ --flat /home/user/Downloads/dss_3chan/flat/ --output /home/user/Downloads/dss_3chan/output/

*/

double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

// controlla file nella directory
int check_directory(const char *dir_path, int *count, long *width, long *height, long *n_chan) {
    DIR *dir = opendir(dir_path);
    if (!dir) { perror("opendir"); return 1; }

    struct dirent *entry;
    int status=0;

    // Conta e misura le immagini
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type != DT_REG) continue;

        if (strstr(entry->d_name, ".fits") || strstr(entry->d_name, ".fit")) {
            char path[1024];
            snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);

            fitsfile *fptr = nullptr;
            open_fits(path, &fptr);
            long w,h,n;
            get_image_dimensions(fptr, &w,&h,&n);
            if (n != 1) {
                fprintf(stderr,"Skipping %s: expected 1 channel\n", path);
                fits_close_file(fptr,&status);
                continue;
            }
            if (*count == 0) { *width=w; *height=h; *n_chan=n; }
            else if (w != *width || h != *height) {
                fprintf(stderr,"Skipping %s: dimensions mismatch\n", path);
                fits_close_file(fptr,&status);
                continue;
            }
            fits_close_file(fptr,&status);
            (*count)++;
        }
    }
    closedir(dir);

    if (*count == 0) { fprintf(stderr,"No valid images\n"); return 1; }
    printf("Found %d images\n", *count);
    return 0;
}

// Rileggi le immagini e copia in memoria chiamando funzione esterna
int load_images_to_memory(const char *dir_path, u_int16_t *img_all, long npixels, int count, cudaMemLocation devLoc) {
    DIR *dir = opendir(dir_path);
    if (!dir) { perror("opendir"); return 1; }

    struct dirent *entry;
    int status=0;
    int idx=0;
    while ((entry = readdir(dir)) != NULL && idx<count) {
        if (entry->d_type != DT_REG) continue;
        if (!(strstr(entry->d_name, ".fits") || strstr(entry->d_name, ".fit"))) continue;

        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);

        fitsfile *fptr = nullptr;
        open_fits(path, &fptr);
        get_fits_data(fptr, npixels, img_all + idx*npixels);
        fits_close_file(fptr,&status);

        CHECK(cudaMemPrefetchAsync(img_all + idx*npixels, npixels*sizeof(u_int16_t), devLoc, 0));
        idx++;
    }
    closedir(dir);
    return 0;
}

int main(int argc, char **argv) {
    const char *in_dir = NULL, *bias_dir = NULL, *dark_dir = NULL, *flat_dir = NULL;
    const char *out_dir = ".";
    const char *bias_name = "MasterBias";
    const char *dark_name = "MasterDark";
    const char *flat_name = "MasterFlat";
    const char *calib_name = "calibrated";

    int opt, option_index = 0;
    static struct option long_options[] = {
        {"bias",  required_argument, 0, 'b'},
        {"dark",  required_argument, 0, 'd'},
        {"flat",  required_argument, 0, 'f'},
        {"light", required_argument, 0, 'l'},
        {"output", required_argument, 0, 'o'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "b:d:f:l:o:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'b': bias_dir = optarg; break;
            case 'd': dark_dir = optarg; break;
            case 'f': flat_dir = optarg; break;
            case 'l': in_dir = optarg; break;
            case 'o': out_dir = optarg; break;
            default:
                fprintf(stderr, "Usage: %s --light <input/dir> --bias <bias/dir> --dark <dark/dir> --flat <flat/dir> [--output <output/dir>]\n", argv[0]);
                return 1;
        }
    }

    if (!in_dir || !bias_dir || !dark_dir || !flat_dir) {
        fprintf(stderr, "Light, bias, dark and flat directories are required.\n");
        return 1;
    }

    remove_trailing_slash((char *)in_dir);
    remove_trailing_slash((char *)bias_dir);
    remove_trailing_slash((char *)dark_dir);
    remove_trailing_slash((char *)flat_dir);

    // Inizializza CUDA
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    CHECK(cudaSetDevice(dev));
    cudaMemLocation devLoc;
    devLoc.id = dev;
    devLoc.type = cudaMemLocationTypeDevice;

    // Controlla file nella directory bias (ritorna numero di immagini e dimensioni)
    long bias_width=0, bias_height=0, bias_n_chan=0;
    int bias_count=0;
    if (check_directory(bias_dir, &bias_count, &bias_width, &bias_height, &bias_n_chan) != 0) {
        fprintf(stderr, "Error checking bias directory\n");
        return 1;
    }

    u_int64_t bias_pixels = bias_width*bias_height;

    // Alloca memoria per unico MasterBias
    u_int16_t *bias_all = nullptr;
    CHECK(cudaMallocManaged(&bias_all, bias_pixels*bias_count*sizeof(u_int16_t)));

    // Rileggi le immagini bias e copia in memoria chiamando funzione esterna
    if (load_images_to_memory(bias_dir, bias_all, bias_pixels, bias_count, devLoc) != 0) {
        fprintf(stderr, "Error loading bias images\n");
        return 1;
    }

    // Alloca memoria per master bias finale (1 immagine)
    u_int16_t *master_bias = nullptr;
    CHECK(cudaMallocManaged(&master_bias, bias_pixels*sizeof(u_int16_t)));
    CHECK(cudaMemPrefetchAsync(master_bias, bias_pixels*sizeof(u_int16_t), devLoc, 0));

    // Calcola master bias
    double t_start = cpuSecond();
    masterBias(bias_all, master_bias, bias_width, bias_height, bias_count);
    double t_elapsed = cpuSecond()-t_start;
    printf("GPU debayer time: %f s\n", t_elapsed);

    // Salva master bias su FITS
    char base_name[128];
    snprintf(base_name, sizeof(base_name), "master_bias");
    save_image_fits(out_dir, base_name, master_bias, bias_width, bias_height, 1);


    // Libera memoria
    CHECK(cudaFree(bias_all));
    CHECK(cudaFree(master_bias));
    CHECK(cudaDeviceReset());

    return 0;
}
