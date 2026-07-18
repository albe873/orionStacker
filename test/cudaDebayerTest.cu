#include "cuda_runtime.h"
#include "common.h"
#include "fits_helper.h"
#include "cuda_helper.h"
#include "debayer.h"

#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <getopt.h>

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
    PrefetchDeviceArg devLoc = make_prefetch_device_arg(dev);

    // =========================
    printf("\n==========================\n");
    printf("Reading input dir\n");

    // Controlla file nella directory
    DIR *dir = opendir(in_dir);
    if (!dir) {
        perror("  opendir");
        exit(1);
    }

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
            get_fits_dimensions(fptr, &w,&h,&n);
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
    printf("  Found %d images\n", image_count);

    u_int64_t npixels = width*height;

    // Alloca memoria continua
    u_int16_t *gray_all = nullptr;
    u_int16_t *rgb_all  = nullptr;
    CHECK(cudaMallocManaged(&gray_all, npixels*image_count*sizeof(u_int16_t)));
    CHECK(cudaMallocManaged(&rgb_all,  npixels*3*image_count*sizeof(u_int16_t)));

    // Rileggi le immagini e copia in memoria
    dir = opendir(in_dir);
    if (!dir) {
        perror("  opendir");
        exit(1);
    }

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
    printf("  Loaded %d images of size %ldx%ld\n", image_count, width, height);

    // ==============================================
    // GPU
    printf("\n==========================\n");
    printf("GPU\n");

    double t_start = cpuSecond();
    demosaic_mhc_rggb(gray_all, rgb_all, width, height, image_count);
    double time_gpu = cpuSecond()-t_start;
    printf("  debayer done - time: %f s\n", time_gpu);

    // ===============================================
    // Saving the images
    printf("\n==========================\n");
    printf("Saving the images\n");
    for (int i = 0; i < image_count; i++) {
        char base_name[128];
        snprintf(base_name, sizeof(base_name), "debayered_%03d", i + 1);
        save_image_fits(out_dir, base_name, rgb_all + i * npixels * 3, width, height, 3);
    }

    // ==============================
    // CPU
    printf("\n==========================\n");
    printf("CPU\n");

    CHECK(cudaMemPrefetchAsync(gray_all, npixels*image_count*sizeof(u_int16_t), cudaCpuDeviceId, 0));
    CHECK(cudaDeviceSynchronize());
    
    // alloca memoria per il risultato CPU
    u_int16_t *rgb_cpu = (u_int16_t *)malloc(npixels*3*image_count*sizeof(u_int16_t));
    if (!rgb_cpu) {
        fprintf(stderr,"Failed to allocate CPU memory\n");
        exit(1);
    }


    t_start = cpuSecond();
    demosaic_mhc_rggb_cpu(gray_all, rgb_cpu, width, height, image_count);
    double time_cpu = cpuSecond()-t_start;
    printf("  debayer done - time: %f s\n", time_cpu);

    // ================================
    // Comparaison
    printf("\n==========================\n");
    printf("Comparing results\n");
    long errors = 0;
    for (int i = 0; i < image_count; i++) {
        for (u_int64_t p = 0; p < npixels*3; p++) {
            u_int16_t gpu_val = rgb_all[i*npixels*3 + p];
            u_int16_t cpu_val = rgb_cpu[i*npixels*3 + p];
            if (gpu_val != cpu_val) {
                errors++;
            }
        }
    }
    if (errors == 0) {
        printf("  GPU and CPU results match!\n");
    } else {
        printf("  GPU and CPU results differ: %ld errors\n", errors);
    }

    // ==================================
    // Performance
    printf("\n==========================\n");
    printf("Performance:\n");
    double speedup = time_cpu / time_gpu;
    printf("  CPU time: %f s,\tGPU time: %f s,\tSpeedup: %f x\n", time_cpu, time_gpu, speedup);



    // Libera memoria
    CHECK(cudaFree(gray_all));
    CHECK(cudaFree(rgb_all));
    free(rgb_cpu);
    CHECK(cudaDeviceReset());

    return 0;
}
