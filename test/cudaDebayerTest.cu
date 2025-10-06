#include "cuda_runtime.h"
#include "fits_api.h"
#include "cuda_check.h"
#include "debayer.h"

#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <ctime>
#include <getopt.h>

double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

int main(int argc, char **argv) {
    const char *in_dir = NULL;
    const char *out_dir = ".";
    const char *file_name = "debayered";

    int opt, option_index = 0;
    static struct option long_options[] = {
        {"input",  required_argument, 0, 'i'},
        {"output", required_argument, 0, 'o'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "i:o:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'i': in_dir = optarg; break;
            case 'o': out_dir = optarg; break;
            default:
                fprintf(stderr, "Usage: %s --input <input/dir> [--output <output/dir>]\n", argv[0]);
                return 1;
        }
    }

    if (!in_dir) {
        fprintf(stderr, "Input directory required.\n");
        return 1;
    }

    remove_trailing_slash((char *)in_dir);

    // Inizializza CUDA
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    CHECK(cudaSetDevice(dev));
    cudaMemLocation devLoc;
    devLoc.id = dev;
    devLoc.type = cudaMemLocationTypeDevice;

    // Controlla file nella directory
    DIR *dir = opendir(in_dir);
    if (!dir) { perror("opendir"); return 1; }

    struct dirent *entry;
    long width=0, height=0, n_chan=0;
    int status=0;
    int image_count=0;

    // Conta e misura le immagini
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type != DT_REG) continue;

        if (strstr(entry->d_name, ".fits") || strstr(entry->d_name, ".fit")) {
            char path[1024];
            snprintf(path, sizeof(path), "%s/%s", in_dir, entry->d_name);

            fitsfile *fptr = nullptr;
            open_fits(path, &fptr);
            long w,h,n;
            get_image_dimensions(fptr, &w,&h,&n);
            if (n != 1) {
                fprintf(stderr,"Skipping %s: expected 1 channel\n", path);
                fits_close_file(fptr,&status);
                continue;
            }
            if (image_count == 0) { width=w; height=h; n_chan=n; }
            else if (w != width || h != height) {
                fprintf(stderr,"Skipping %s: dimensions mismatch\n", path);
                fits_close_file(fptr,&status);
                continue;
            }
            fits_close_file(fptr,&status);
            image_count++;
        }
    }
    closedir(dir);

    if (image_count == 0) { fprintf(stderr,"No valid images\n"); return 1; }
    printf("Found %d images\n", image_count);

    u_int64_t npixels = width*height;

    // Alloca memoria continua
    u_int16_t *gray_all = nullptr;
    u_int16_t *rgb_all  = nullptr;
    CHECK(cudaMallocManaged(&gray_all, npixels*image_count*sizeof(u_int16_t)));
    CHECK(cudaMallocManaged(&rgb_all,  npixels*3*image_count*sizeof(u_int16_t)));

    // Rileggi le immagini e copia in memoria
    dir = opendir(in_dir);
    if (!dir) { perror("opendir"); return 1; }

    int idx=0;
    while ((entry = readdir(dir)) != NULL && idx<image_count) {
        if (entry->d_type != DT_REG) continue;
        if (!(strstr(entry->d_name, ".fits") || strstr(entry->d_name, ".fit"))) continue;

        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", in_dir, entry->d_name);

        fitsfile *fptr = nullptr;
        open_fits(path, &fptr);
        get_fits_data(fptr, npixels, gray_all + idx*npixels);
        fits_close_file(fptr,&status);

        CHECK(cudaMemPrefetchAsync(gray_all + idx*npixels, npixels*sizeof(u_int16_t), devLoc, 0));
        idx++;
    }
    closedir(dir);

    CHECK(cudaMemPrefetchAsync(rgb_all, npixels*3*image_count*sizeof(u_int16_t), devLoc, 0));

    // Esegui kernel

    double t_start = cpuSecond();
    //demosaic_bilinear_rggb_kernel<<<grid_size, block_size>>>(gray_all,rgb_all,width,height,image_count);
    demosaic_mhc_rggb(gray_all, rgb_all, width, height, image_count);
    double t_elapsed = cpuSecond()-t_start;
    printf("GPU debayer time: %f s\n", t_elapsed);

    // Salva immagini RGB
    for (int i = 0; i < image_count; i++) {
        char base_name[128];
        snprintf(base_name, sizeof(base_name), "debayered_%03d", i + 1);
        save_image_fits(out_dir, base_name, rgb_all + i * npixels * 3, width, height, 3);
    }


    // Libera memoria
    CHECK(cudaFree(gray_all));
    CHECK(cudaFree(rgb_all));
    CHECK(cudaDeviceReset());

    return 0;
}
