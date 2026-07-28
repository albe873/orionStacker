#include "fits_helper.hh"
#include "cuda_helper.hh"
#include "common.hh"

#include "stacker.hh"

#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <cstring>

int main(int argc, char **argv) {

    // ==================================================
    // Parsing degli argomenti
    const char *in_dir = NULL;
    const char *out_dir = ".";
    const char *file_name = "image";

    float kappa = 3.0f;
    uint16_t iterations = 5;

    int opt, option_index = 0;
    static struct option long_options[] = {
        {"input-directory",  required_argument, 0, 'i'},
        {"output-directory", optional_argument, 0, 'o'},
        {"file-name",        optional_argument, 0, 'n'},
        {"kappa",            optional_argument, 0, 'k'},
        {"iterations",           optional_argument, 0, 't'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "i:o:n:k:s:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'i': in_dir   = optarg; break;
            case 'o': out_dir  = optarg; break;
            case 'n': file_name = optarg; break;
            case 'k': {
                char *end;
                float v = strtof(optarg, &end);
                if (end != optarg && v >= 0.0f && v <= 100.0f) kappa = v;
                else fprintf(stderr, "Invalid kappa, using default %.1f\n", kappa);
                break;
            }
            case 't': {
                char *end;
                long v = strtol(optarg, &end, 10);
                if (end != optarg && v >= 0 && v <= 65535) iterations = (u_int16_t)v;
                else fprintf(stderr, "Invalid iterations, using default %d\n", iterations);
                break;
            }
            default:
                fprintf(stderr, "Usage: %s --input-directory <dir> [--output-directory <dir>] "
                                "[--file-name <name>] [--kappa <f>] [--iterations <n>]\n", argv[0]);
                return 1;
        }
    }

    if (!in_dir) {
        fprintf(stderr, "Input directory required.\n");
        return 1;
    }

    // ==================================================
    // Inizializza CUDA
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    CHECK(cudaSetDevice(dev));
    PrefetchDeviceArg devLoc = make_prefetch_device_arg(dev);

    // ==================================================
    // Controlla directory e ottiene dimensioni / conteggio
    remove_trailing_slash((char *)in_dir);

    long width = 0, height = 0, n_chan = 0;
    int image_count = 0;

    printf("\n==========================\n");
    printf("Reading directory\n");

    if (check_directory(in_dir, &image_count, &width, &height, &n_chan, 0) != 0) {
        fprintf(stderr, "Error checking input directory\n");
        return 1;
    }

    uint64_t npixels = (uint64_t)width * height * n_chan;

    if (image_count == 0) {
        fprintf(stderr, "No valid FITS images found in %s\n", in_dir);
        return 1;
    }
    if (image_count == 1) {
        fprintf(stderr, "Only one image found, no stacking needed\n");
        return 0;
    }

    printf("  Images of size %ldx%ld, %ld channel(s)\n", width, height, n_chan);

    // Allocazione memoria GPU
    uint16_t *img_all = nullptr;
    CHECK(cudaMallocManaged(&img_all, (size_t)npixels * image_count * sizeof(uint16_t)));

    // Carica le immagini in memoria
    printf("  Loading images ...\n");
    if (load_images_to_memory_prefetch(in_dir, img_all, width, height, n_chan, image_count, dev) != 0) {
        fprintf(stderr, "Error loading images\n");
        return 1;
    }
    printf("  Loaded %d images\n", image_count);

    // Alloca memoria per il risultato GPU
    uint16_t *mean_gpu = nullptr;
    CHECK(cudaMallocManaged(&mean_gpu, (size_t)npixels * sizeof(uint16_t)));
    CHECK(cudaMemAdvise(mean_gpu, (size_t)npixels * sizeof(uint16_t), cudaMemAdviseSetPreferredLocation, devLoc));

    // ==================================================
    // GPU Alfa Sigma
    printf("\n==========================\n");
    printf("GPU\n");

    double t_start = cpuSecond();
    winsorized_sigma_clipping_gpu(img_all, mean_gpu, (uint16_t)image_count, npixels, kappa);
    double time_gpu = cpuSecond() - t_start;
    printf("  Alfa Sigma elapsed time: %f s\n", time_gpu);

    // ==================================================
    // Salva risultato GPU
    printf("\n==========================\n");
    printf("Saving GPU result\n");
    save_image_fits(out_dir, file_name, mean_gpu, width, height, n_chan);
    // save_image_tiff(out_dir, file_name, mean_gpu, width, height, n_chan);

    // ==================================================
    // CPU Alfa Sigma
    // IMPORTANTE: Rileggiamo le immagini perché la GPU le ha modificate
    printf("\n==========================\n");
    printf("CPU\n");

    // Alloca buffer flat per CPU e rilegge le immagini dal disco
    uint16_t *img_all_cpu = (uint16_t *)malloc((size_t)npixels * image_count * sizeof(uint16_t));
    if (!img_all_cpu) {
        fprintf(stderr, "Failed to allocate CPU memory\n");
        return 1;
    }

    if (load_images_to_memory(in_dir, img_all_cpu, width, height, n_chan, image_count) != 0) {
        fprintf(stderr, "Error loading images for CPU\n");
        free(img_all_cpu);
        return 1;
    }

    uint16_t *mean_cpu = (uint16_t *)malloc((size_t)npixels * sizeof(uint16_t));
    if (!mean_cpu) {
        fprintf(stderr, "Failed to allocate CPU result memory\n");
        free(img_all_cpu); free(mean_cpu);
        return 1;
    }

    t_start = cpuSecond();
    winsorized_sigma_clipping_cpu(img_all_cpu, mean_cpu, image_count, (int)npixels, kappa);
    double time_cpu = cpuSecond() - t_start;
    printf("  CPU Alfa Sigma elapsed time: %f s\n", time_cpu);

    // ==================================================
    // Comparazione risultati
    printf("\n==========================\n");
    printf("Comparing results\n");

    long errors = 0;
    for (uint64_t i = 0; i < npixels; i++) {
        if (mean_gpu[i] != mean_cpu[i])
            errors++;
    }

    if (errors == 0) {
        printf("  GPU and CPU results match!\n");
    } else {
        printf("  GPU and CPU results differences: %ld / %lu pixels mismatch (%.4f%%)\n",
               errors, (unsigned long)npixels, 100.0 * errors / npixels);
        // Mostra alcuni dettagli
        long shown = 0;
        for (uint64_t i = 0; i < npixels && shown < 10; i++) {
            if (mean_gpu[i] != mean_cpu[i]) {
                printf("    pixel[%lu]: GPU=%u  CPU=%u\n",
                       (unsigned long)i, (unsigned)mean_gpu[i], (unsigned)mean_cpu[i]);
                shown++;
            }
        }
    }

    // ==================================================
    // Performance
    printf("\n==========================\n");
    printf("Performance:\n");
    double speedup = time_cpu / time_gpu;
    printf("  CPU time: %f s,\tGPU time: %f s,\tSpeedup: %f x\n", time_cpu, time_gpu, speedup);

    // ==================================================
    // Cleanup
    CHECK(cudaFree(img_all));
    CHECK(cudaFree(mean_gpu));
    free(img_all_cpu);
    free(mean_cpu);

    CHECK(cudaDeviceReset());
    return 0;
}
