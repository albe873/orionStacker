#include "cuda_runtime.h"
#include "common.h"
#include "fits_helper.h"
#include "cuda_helper.h"
#include "debayer.h"

#include <stdio.h>
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
    long width=0, height=0, n_chan=0;
    int image_count=0;

    if (check_directory(in_dir, &image_count, &width, &height, &n_chan, 1) != 0) {
        return 1;
    }

    u_int64_t npixels = width*height;

    // Alloca memoria continua
    u_int16_t *gray_all = nullptr;
    u_int16_t *rgb_all  = nullptr;
    CHECK(cudaMallocManaged(&gray_all, npixels*image_count*sizeof(u_int16_t)));
    CHECK(cudaMallocManaged(&rgb_all,  npixels*3*image_count*sizeof(u_int16_t)));
    CHECK(cudaMemPrefetchAsync(rgb_all, npixels*3*image_count*sizeof(u_int16_t), devLoc, 0));

    // Rileggi le immagini e copia in memoria
    if (load_images_to_memory_prefetch(in_dir, gray_all, width, height, n_chan, image_count, dev) != 0) {
        fprintf(stderr, "Error loading images\n");
        return 1;
    }

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
