#include "fits_helper.hh"
#include "cuda_helper.hh"
#include "common.hh"

#include "calibration.hh"

#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <algorithm>

using namespace std;

/* 

eseguire con:
./cudaCalibration --light /home/user/Downloads/dss_3chan/input/ --bias /home/user/Downloads/dss_3chan/bias --dark /home/user/Downloads/dss_3chan/dark/ --flat /home/user/Downloads/dss_3chan/flat/ --output /home/user/Downloads/dss_3chan/output/

*/

int main(int argc, char **argv) {
    const char *in_dir = NULL, *bias_dir = NULL, *dark_dir = NULL, *flat_dir = NULL;
    const char *out_dir = ".";

    // Persistent strings to hold dynamically constructed paths
    std::string base_dir_str, bias_dir_str, dark_dir_str, flat_dir_str, in_dir_str;

    int opt, option_index = 0;
    static struct option long_options[] = {
        {"base-dir", required_argument, 0, 'B'},
        {"bias",  required_argument, 0, 'b'},
        {"dark",  required_argument, 0, 'd'},
        {"flat",  required_argument, 0, 'f'},
        {"light", required_argument, 0, 'l'},
        {"output", required_argument, 0, 'o'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "b:d:f:l:o:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'B': {
                // If base directory is provided, append subdirectories for bias, dark, flat, and light
                remove_trailing_slash(optarg);
                base_dir_str = optarg;
                bias_dir_str = base_dir_str + "/bias";
                dark_dir_str = base_dir_str + "/dark";
                flat_dir_str = base_dir_str + "/flat";
                in_dir_str = base_dir_str + "/light";
                bias_dir = bias_dir_str.c_str();
                dark_dir = dark_dir_str.c_str();
                flat_dir = flat_dir_str.c_str();
                in_dir = in_dir_str.c_str();
                break;
            }
            case 'b': bias_dir = optarg; remove_trailing_slash((char *)bias_dir); break;
            case 'd': dark_dir = optarg; remove_trailing_slash((char *)dark_dir); break;
            case 'f': flat_dir = optarg; remove_trailing_slash((char *)flat_dir); break;
            case 'l': in_dir = optarg; remove_trailing_slash((char *)in_dir); break;
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

    // Inizializza CUDA
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    CHECK(cudaSetDevice(dev));
    PrefetchDeviceArg devLoc = make_prefetch_device_arg(dev);

    // ==================================================
    // GPU Calibration
    printf("\n==========================================\n");
    printf("Calibration in GPU\n\n");

    /************************************************** 1) MASTER BIAS **************************************************/
    // Controlla file nella directory bias (ritorna numero di immagini e dimensioni)
    long bias_width=0, bias_height=0, bias_n_chan=0;
    int bias_count=0;
    if (check_directory(bias_dir, &bias_count, &bias_width, &bias_height, &bias_n_chan, 1) != 0) {
        fprintf(stderr, "Error checking bias directory\n");
        return 1;
    }

    uint64_t bias_pixels = bias_width*bias_height;

    // Alloca memoria bias
    uint16_t *bias_all = nullptr;
    CHECK(cudaMallocManaged(&bias_all, bias_pixels*bias_count*sizeof(uint16_t)));

    // Rileggi le immagini bias e copia in memoria chiamando funzione esterna
    if (load_images_to_memory_prefetch(bias_dir, bias_all, bias_width, bias_height, bias_n_chan, bias_count, dev) != 0) {
        fprintf(stderr, "Error loading bias images\n");
        return 1;
    }

    // Alloca memoria per master bias finale (1 immagine)
    float *master_bias = nullptr;
    CHECK(cudaMallocManaged(&master_bias, bias_pixels*sizeof(float)));
    CHECK(cudaMemPrefetchAsync(master_bias, bias_pixels*sizeof(float), devLoc, 0));

    // Calcola master bias
    double t_start = cpuSecond();
    masterBias_gpu(bias_all, master_bias, bias_width, bias_height, bias_count);
    double time_master_bias_gpu = cpuSecond()-t_start;
    printf("  GPU master bias time: %f s\n", time_master_bias_gpu);

    /************************************************** 2) MASTER DARK **************************************************/
    // controlla file nella directory dark (ritorna numero di immagini e dimensioni)
    long dark_width=0, dark_height=0, dark_n_chan=0;
    int dark_count=0;
    if (check_directory(dark_dir, &dark_count, &dark_width, &dark_height, &dark_n_chan, 1) != 0) {
        fprintf(stderr, "Error checking dark directory\n");
        return 1;
    }

    uint64_t dark_pixels = dark_width*dark_height;

    // Alloca memoria per dark
    uint16_t *dark_all = nullptr;
    CHECK(cudaMallocManaged(&dark_all, dark_pixels*dark_count*sizeof(uint16_t)));

    // Rileggi le immagini dark e copia in memoria chiamando funzione esterna
    if (load_images_to_memory_prefetch(dark_dir, dark_all, dark_width, dark_height, dark_n_chan, dark_count, dev) != 0) {
        fprintf(stderr, "Error loading dark images\n");
        return 1;
    }

    // Alloca memoria per master dark finale (1 immagine)
    float *master_dark = nullptr;
    CHECK(cudaMallocManaged(&master_dark, dark_pixels*sizeof(float)));
    CHECK(cudaMemPrefetchAsync(master_dark, dark_pixels*sizeof(float), devLoc, 0));

    //calcola master dark
    t_start = cpuSecond();
    masterDark_gpu(dark_all, master_bias, master_dark, dark_width, dark_height, dark_count);
    double time_master_dark_gpu = cpuSecond()-t_start;
    printf("  GPU master dark time: %f s\n", time_master_dark_gpu);

    /************************************************** 3) MASTER FLAT **************************************************/
    // controlla file nella directory flat (ritorna numero di immagini e dimensioni)
    long flat_width=0, flat_height=0, flat_n_chan=0;
    int flat_count=0;
    if (check_directory(flat_dir, &flat_count, &flat_width, &flat_height, &flat_n_chan, 1) != 0) {
        fprintf(stderr, "Error checking flat directory\n");
        return 1;
    }

    uint64_t flat_pixels = flat_width*flat_height;

    // Alloca memoria per flat
    uint16_t *flat_all = nullptr;
    CHECK(cudaMallocManaged(&flat_all, flat_pixels*flat_count*sizeof(uint16_t)));

    // Rileggi le immagini flat e copia in memoria chiamando funzione esterna
    if (load_images_to_memory_prefetch(flat_dir, flat_all, flat_width, flat_height, flat_n_chan, flat_count, dev) != 0) {
        fprintf(stderr, "Error loading flat images\n");
        return 1;
    }

    // Alloca memoria per master flat finale (1 immagine)
    float *master_flat = nullptr;
    CHECK(cudaMallocManaged(&master_flat, flat_pixels*sizeof(float)));
    CHECK(cudaMemPrefetchAsync(master_flat, flat_pixels*sizeof(float), devLoc, 0));

    //calcola master flat
    t_start = cpuSecond();
    masterFlat_gpu(flat_all, master_bias, master_flat, flat_width, flat_height, flat_count);
    double time_master_flat_gpu = cpuSecond()-t_start;
    printf("  GPU master flat time: %f s\n", time_master_flat_gpu);

    /************************************************** 4) CALIBRATED LIGHT **************************************************/
    // controlla file nella directory light (ritorna numero di immagini e dimensioni)
    long light_width=0, light_height=0, light_n_chan=0;
    int light_count=0;
    if (check_directory(in_dir, &light_count, &light_width, &light_height, &light_n_chan, 1) != 0) {
        fprintf(stderr, "Error checking light directory\n");
        return 1;
    }

    uint64_t light_pixels = light_width*light_height;

    // Alloca memoria per light
    uint16_t *light_all = nullptr;
    CHECK(cudaMallocManaged(&light_all, light_pixels*light_count*sizeof(uint16_t)));

    // Rileggi le immagini light e copia in memoria chiamando funzione esterna
    if (load_images_to_memory_prefetch(in_dir, light_all, light_width, light_height, light_n_chan, light_count, dev) != 0) {
        fprintf(stderr, "Error loading light images\n");
        return 1;
    }

    // Alloca memoria per immagini calibrate finali (light_count immagini)
    uint16_t *calib_all = nullptr;
    CHECK(cudaMallocManaged(&calib_all, light_pixels*light_count*sizeof(uint16_t)));
    CHECK(cudaMemPrefetchAsync(calib_all, light_pixels*light_count*sizeof(uint16_t), devLoc, 0));

    //calibra immagini light
    t_start = cpuSecond();
    calibrateLights_gpu(light_all, master_bias, master_dark, master_flat, calib_all, light_width, light_height, light_count);
    double time_calibrate_lights_gpu = cpuSecond()-t_start;
    printf("  GPU calibrate lights time: %f s\n", time_calibrate_lights_gpu);



    // ==================================================
    // CPU Calibration
    printf("\n==========================================\n");
    printf("Calibration in CPU\n\n");

    // Prefetch bias_all, dark_all, flat_all, light_all to CPU
    auto pf_host_id = make_prefetch_host_arg();
    CHECK(cudaMemPrefetchAsync(bias_all, bias_pixels*bias_count*sizeof(uint16_t), pf_host_id, 0));
    CHECK(cudaMemPrefetchAsync(dark_all, dark_pixels*dark_count*sizeof(uint16_t), pf_host_id, 0));
    CHECK(cudaMemPrefetchAsync(flat_all, flat_pixels*flat_count*sizeof(uint16_t), pf_host_id, 0));
    CHECK(cudaMemPrefetchAsync(light_all, light_pixels*light_count*sizeof(uint16_t), pf_host_id, 0));
    // Wait for prefetch to complete before accessing memory
    CHECK(cudaDeviceSynchronize());

    /************************************************** 1) MASTER BIAS **************************************************/
    // Controlla file nella directory bias (ritorna numero di immagini e dimensioni)
    // Alloca memoria bias

    // Alloca memoria per master bias finale (1 immagine)
    float *master_bias_cpu = (float*)malloc(bias_pixels*sizeof(float));
    if (!master_bias_cpu) {
        fprintf(stderr, "Error allocating memory for master bias CPU\n");
        return 1;
    }

    // Calcola master bias
    t_start = cpuSecond();
    masterBias_cpu(bias_all, master_bias_cpu, bias_width, bias_height, bias_count);
    double time_master_bias_cpu = cpuSecond()-t_start;
    printf("  CPU master bias time: %f s\n", time_master_bias_cpu); 


    /************************************************** 2) MASTER DARK **************************************************/

    // Alloca memoria per master dark finale (1 immagine)
    float *master_dark_cpu = (float*)malloc(dark_pixels*sizeof(float));
    if (!master_dark_cpu) {
        fprintf(stderr, "Error allocating memory for master dark CPU\n");
        return 1;
    }  

    //calcola master dark
    t_start = cpuSecond();
    masterDark_cpu(dark_all, master_bias_cpu, master_dark_cpu, dark_width, dark_height, dark_count);
    double time_master_dark_cpu = cpuSecond()-t_start;
    printf("  CPU master dark time: %f s\n", time_master_dark_cpu);


    /************************************************** 3) MASTER FLAT **************************************************/

    // Alloca memoria per master flat finale (1 immagine)
    float *master_flat_cpu = (float*)malloc(flat_pixels*sizeof(float));
    if (!master_flat_cpu) {
        fprintf(stderr, "Error allocating memory for master flat CPU\n");
        return 1;
    }

    //calcola master flat
    t_start = cpuSecond();
    masterFlat_cpu(flat_all, master_bias_cpu, master_flat_cpu, flat_width, flat_height, flat_count);
    double time_master_flat_cpu = cpuSecond()-t_start;
    printf("  CPU master flat time: %f s\n", time_master_flat_cpu);


    /************************************************** 4) CALIBRATED LIGHT **************************************************/

    // Alloca memoria per immagini calibrate finali (light_count immagini)
    uint16_t *calib_all_cpu = (uint16_t *)malloc(light_pixels*light_count*sizeof(uint16_t));
    if (!calib_all_cpu) {
        fprintf(stderr, "Error allocating memory for calibrated images CPU\n");
        return 1;
    }

    //calibra immagini light
    t_start = cpuSecond();
    calibrateLights_cpu(light_all, master_bias_cpu, master_dark_cpu, master_flat_cpu, calib_all_cpu, light_width, light_height, light_count);
    double time_calibrate_lights_cpu = cpuSecond()-t_start;
    printf("  CPU calibrate lights time: %f s\n", time_calibrate_lights_cpu);



    // ==================================================
    // Comparazione risultati
    printf("\n==========================================\n");
    printf("Comparing GPU and CPU results\n\n");

    // 1. Comparazione master bias
    int match = 0, mismatch = 0;
    for (int i = 0; i < bias_pixels; i++) {
        if (master_bias[i] == master_bias_cpu[i])
            match++;
        else
            mismatch++;
    }
    printf("  Master Bias: \t\t%d matches, \t%d mismatches\n", match, mismatch);

    // 2. Comparazione master dark
    match = 0; mismatch = 0;
    for (int i = 0; i < dark_pixels; i++) {
        if (master_dark[i] == master_dark_cpu[i])
            match++;
        else
            mismatch++;
    }
    printf("  Master Dark: \t\t%d matches, \t%d mismatches\n", match, mismatch);
    
    // 3. Comparazione master flat
    match = 0; mismatch = 0;
    for (int i = 0; i < flat_pixels; i++) {
        if (master_flat[i] == master_flat_cpu[i])
            match++;
        else
            mismatch++;
    }
    printf("  Master Flat: \t\t%d matches, \t%d mismatches\n", match, mismatch);
    
    // 4. Comparazione immagini calibrate
    match = 0; mismatch = 0;
    for (int i = 0; i < light_count*light_pixels; i++) {
        if (calib_all[i] == calib_all_cpu[i])
            match++;
        else
            mismatch++;
    }
    printf("  Calibrated Images: \t%d matches, \t%d mismatches\n", match, mismatch);

    
    // ==================================================
    // Performance
    printf("\n==========================================\n");
    printf("Performance Comparison\n\n");

    double speedup_master_bias = time_master_bias_cpu / time_master_bias_gpu;
    double speedup_master_dark = time_master_dark_cpu / time_master_dark_gpu;
    double speedup_master_flat = time_master_flat_cpu / time_master_flat_gpu;
    double speedup_calibrate_lights = time_calibrate_lights_cpu / time_calibrate_lights_gpu;

    printf("  Master Bias: \t\tGPU time = %f s, \tCPU time = %f s, \tSpeedup = %.1f x\n", time_master_bias_gpu, time_master_bias_cpu, speedup_master_bias);
    printf("  Master Dark: \t\tGPU time = %f s, \tCPU time = %f s, \tSpeedup = %.1f x\n", time_master_dark_gpu, time_master_dark_cpu, speedup_master_dark);
    printf("  Master Flat: \t\tGPU time = %f s, \tCPU time = %f s, \tSpeedup = %.1f x\n", time_master_flat_gpu, time_master_flat_cpu, speedup_master_flat);
    printf("  Calibrate Lights: \tGPU time = %f s, \tCPU time = %f s, \tSpeedup = %.1f x\n\n", time_calibrate_lights_gpu, time_calibrate_lights_cpu, speedup_calibrate_lights);

    double time_total_gpu = time_master_bias_gpu + time_master_dark_gpu + time_master_flat_gpu + time_calibrate_lights_gpu;
    double time_total_cpu = time_master_bias_cpu + time_master_dark_cpu + time_master_flat_cpu + time_calibrate_lights_cpu;
    double speedup_total = time_total_cpu / time_total_gpu;
    printf("  Total: \t\tGPU time = %f s, \tCPU time = %f s, \tSpeedup = %.1f x\n\n\n", time_total_gpu, time_total_cpu, speedup_total);

    // ==================================================
    // Salva immagini FITS

    // Salva master bias su FITS
    char base_name[128];
    snprintf(base_name, sizeof(base_name), "master_bias");
    save_image_fits(out_dir, base_name, master_bias, bias_width, bias_height, 1);

    // Salva master dark su FITS
    snprintf(base_name, sizeof(base_name), "master_dark");
    save_image_fits(out_dir, base_name, master_dark, dark_width, dark_height, 1);

    // Salva master flat su FITS
    snprintf(base_name, sizeof(base_name), "master_flat");
    save_image_fits(out_dir, base_name, master_flat, flat_width, flat_height, 1);

    // Salva immagini calibrate su FITS
    for (int i = 0; i < light_count; i++) {
        snprintf(base_name, sizeof(base_name), "calibrated_%d", i);
        save_image_fits(out_dir, base_name, &calib_all[i*light_pixels], light_width, light_height, 1);
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