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
int load_images_to_memory(const char *dir_path, float *img_all, long npixels, int count, cudaMemLocation devLoc) {
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
        u_int16_t *temp = (u_int16_t*)malloc(npixels * sizeof(u_int16_t));
        get_fits_data(fptr, npixels, temp);
        for (long i = 0; i < npixels; i++) {
            img_all[idx * npixels + i] = (float)temp[i];
        }
        free(temp);
        fits_close_file(fptr,&status);

        CHECK(cudaMemPrefetchAsync(img_all + idx*npixels, npixels*sizeof(float), devLoc, 0));
        idx++;
    }
    closedir(dir);
    return 0;
}

int main(int argc, char **argv) {
    const char *in_dir = NULL, *bias_dir = NULL, *dark_dir = NULL, *flat_dir = NULL;
    const char *out_dir = ".";

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

    /************************************************** 1) MASTER BIAS **************************************************/
    // Controlla file nella directory bias (ritorna numero di immagini e dimensioni)
    long bias_width=0, bias_height=0, bias_n_chan=0;
    int bias_count=0;
    if (check_directory(bias_dir, &bias_count, &bias_width, &bias_height, &bias_n_chan) != 0) {
        fprintf(stderr, "Error checking bias directory\n");
        return 1;
    }

    u_int64_t bias_pixels = bias_width*bias_height;

    // Alloca memoria bias
    float *bias_all = nullptr;
    CHECK(cudaMallocManaged(&bias_all, bias_pixels*bias_count*sizeof(float)));

    // Rileggi le immagini bias e copia in memoria chiamando funzione esterna
    if (load_images_to_memory(bias_dir, bias_all, bias_pixels, bias_count, devLoc) != 0) {
        fprintf(stderr, "Error loading bias images\n");
        return 1;
    }

    // Alloca memoria per master bias finale (1 immagine)
    float *master_bias = nullptr;
    CHECK(cudaMallocManaged(&master_bias, bias_pixels*sizeof(float)));
    CHECK(cudaMemPrefetchAsync(master_bias, bias_pixels*sizeof(float), devLoc, 0));

    // Calcola master bias
    double t_start = cpuSecond();
    masterBias(bias_all, master_bias, bias_width, bias_height, bias_count);
    double t_elapsed = cpuSecond()-t_start;
    printf("GPU debayer time: %f s\n", t_elapsed);

    // Salva master bias su FITS
    char base_name[128];
    snprintf(base_name, sizeof(base_name), "master_bias");
    u_int16_t *temp_bias = (u_int16_t*)malloc(bias_pixels * sizeof(u_int16_t));
    for (u_int64_t i = 0; i < bias_pixels; i++) {
        temp_bias[i] = (u_int16_t)fminf(fmaxf(master_bias[i], 0.0f), 65535.0f);
    }
    save_image_fits(out_dir, base_name, temp_bias, bias_width, bias_height, 1);
    free(temp_bias);

    /************************************************** 2) MASTER DARK **************************************************/
    // controlla file nella directory dark (ritorna numero di immagini e dimensioni)
    long dark_width=0, dark_height=0, dark_n_chan=0;
    int dark_count=0;
    if (check_directory(dark_dir, &dark_count, &dark_width, &dark_height, &dark_n_chan) != 0) {
        fprintf(stderr, "Error checking dark directory\n");
        return 1;
    }

    u_int64_t dark_pixels = dark_width*dark_height;

    // Alloca memoria per dark
    float *dark_all = nullptr;
    CHECK(cudaMallocManaged(&dark_all, dark_pixels*dark_count*sizeof(float)));

    // Rileggi le immagini dark e copia in memoria chiamando funzione esterna
    if (load_images_to_memory(dark_dir, dark_all, dark_pixels, dark_count, devLoc) != 0) {
        fprintf(stderr, "Error loading dark images\n");
        return 1;
    }

    // Alloca memoria per master dark finale (1 immagine)
    float *master_dark = nullptr;
    CHECK(cudaMallocManaged(&master_dark, dark_pixels*sizeof(float)));
    CHECK(cudaMemPrefetchAsync(master_dark, dark_pixels*sizeof(float), devLoc, 0));

    //calcola master dark
    t_start = cpuSecond();
    masterDark(dark_all, master_bias, master_dark, dark_width, dark_height, dark_count);
    t_elapsed = cpuSecond()-t_start;
    printf("GPU master dark time: %f s\n", t_elapsed);

    // Salva master dark su FITS
    snprintf(base_name, sizeof(base_name), "master_dark");
    u_int16_t *temp_dark = (u_int16_t*)malloc(dark_pixels * sizeof(u_int16_t));
    for (u_int64_t i = 0; i < dark_pixels; i++) {
        temp_dark[i] = (u_int16_t)fminf(fmaxf(master_dark[i], 0.0f), 65535.0f);
    }
    save_image_fits(out_dir, base_name, temp_dark, dark_width, dark_height, 1);
    free(temp_dark);

    /************************************************** 3) MASTER FLAT **************************************************/
    // controlla file nella directory flat (ritorna numero di immagini e dimensioni)
    long flat_width=0, flat_height=0, flat_n_chan=0;
    int flat_count=0;
    if (check_directory(flat_dir, &flat_count, &flat_width, &flat_height, &flat_n_chan) != 0) {
        fprintf(stderr, "Error checking flat directory\n");
        return 1;
    }

    u_int64_t flat_pixels = flat_width*flat_height;

    // Alloca memoria per flat
    float *flat_all = nullptr;
    CHECK(cudaMallocManaged(&flat_all, flat_pixels*flat_count*sizeof(float)));

    // Rileggi le immagini flat e copia in memoria chiamando funzione esterna
    if (load_images_to_memory(flat_dir, flat_all, flat_pixels, flat_count, devLoc) != 0) {
        fprintf(stderr, "Error loading flat images\n");
        return 1;
    }

    // Alloca memoria per master flat finale (1 immagine)
    float *master_flat = nullptr;
    CHECK(cudaMallocManaged(&master_flat, flat_pixels*sizeof(float)));
    CHECK(cudaMemPrefetchAsync(master_flat, flat_pixels*sizeof(float), devLoc, 0));

    //calcola master flat
    t_start = cpuSecond();
    masterFlat(flat_all, master_bias, master_flat, flat_width, flat_height, flat_count);
    t_elapsed = cpuSecond()-t_start;
    printf("GPU master flat time: %f s\n", t_elapsed);

    // Salva master flat su FITS
    snprintf(base_name, sizeof(base_name), "master_flat");
    u_int16_t *temp_flat = (u_int16_t*)malloc(flat_pixels * sizeof(u_int16_t));
    for (u_int64_t i = 0; i < flat_pixels; i++) {
        temp_flat[i] = (u_int16_t)fminf(fmaxf(master_flat[i], 0.0f), 65535.0f);
    }
    save_image_fits(out_dir, base_name, temp_flat, flat_width, flat_height, 1);
    free(temp_flat);

    /************************************************** 4) CALIBRATED LIGHT **************************************************/
    // controlla file nella directory light (ritorna numero di immagini e dimensioni)
    long light_width=0, light_height=0, light_n_chan=0;
    int light_count=0;
    if (check_directory(in_dir, &light_count, &light_width, &light_height, &light_n_chan) != 0) {
        fprintf(stderr, "Error checking light directory\n");
        return 1;
    }

    u_int64_t light_pixels = light_width*light_height;

    // Alloca memoria per light
    float *light_all = nullptr;
    CHECK(cudaMallocManaged(&light_all, light_pixels*light_count*sizeof(float)));

    // Rileggi le immagini light e copia in memoria chiamando funzione esterna
    if (load_images_to_memory(in_dir, light_all, light_pixels, light_count, devLoc) != 0) {
        fprintf(stderr, "Error loading light images\n");
        return 1;
    }

    // Alloca memoria per immagini calibrate finali (light_count immagini)
    float *calib_all = nullptr;
    CHECK(cudaMallocManaged(&calib_all, light_pixels*light_count*sizeof(float)));
    CHECK(cudaMemPrefetchAsync(calib_all, light_pixels*light_count*sizeof(float), devLoc, 0));

    //calibra immagini light
    t_start = cpuSecond();
    calibrateLights(light_all, master_bias, master_dark, master_flat, calib_all, light_width, light_height, light_count);
    t_elapsed = cpuSecond()-t_start;
    printf("GPU calibrate lights time: %f s\n", t_elapsed);

    // Salva immagini calibrate su FITS
    for (int i = 0; i < light_count; i++) {
        snprintf(base_name, sizeof(base_name), "calibrated_%d", i);
        u_int16_t *temp_calib = (u_int16_t*)malloc(light_pixels * sizeof(u_int16_t));
        for (u_int64_t j = 0; j < light_pixels; j++) {
            temp_calib[j] = (u_int16_t)fminf(fmaxf(calib_all[i*light_pixels + j], 0.0f), 65535.0f);
        }
        save_image_fits(out_dir, base_name, temp_calib, light_width, light_height, 1);
        free(temp_calib);
    }

    // Libera memoria
    CHECK(cudaFree(bias_all));
    CHECK(cudaFree(master_bias));
    CHECK(cudaFree(dark_all));
    CHECK(cudaFree(master_dark));
    CHECK(cudaFree(flat_all));
    CHECK(cudaFree(master_flat));
    CHECK(cudaFree(light_all));
    CHECK(cudaFree(calib_all));
    CHECK(cudaDeviceReset());

    return 0;
}
