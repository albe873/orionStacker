#include "cuda_helper.hh"
#include "fits_helper.hh"
#include "common.hh"

#include "calibration.hh"
#include "debayer.hh"
#include "star_finder.hh"
#include "stacker.hh"

#include <getopt.h>
#include <unistd.h>
#include <string>
#include <cstring>

int main(int argc, char** argv) {
    const char *in_path = nullptr, *bias_path = nullptr, *dark_path = nullptr, *flat_path = nullptr;
    const char *out_path = ".";

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

    while ((opt = getopt_long(argc, argv, "B:b:d:f:l:o:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'B': {
                // If base directory is provided, append subdirectories for bias, dark, flat, and light
                remove_trailing_slash(optarg);
                static std::string s_bias, s_dark, s_flat, s_in, s_out;
                s_bias = std::string(optarg) + "/bias";
                s_dark = std::string(optarg) + "/dark";
                s_flat = std::string(optarg) + "/flat";
                s_in   = std::string(optarg) + "/light";
                s_out  = std::string(optarg) + "/output";
                bias_path = s_bias.c_str();
                dark_path = s_dark.c_str();
                flat_path = s_flat.c_str();
                in_path   = s_in.c_str();
                out_path  = s_out.c_str();
                break;
            }
            case 'b': bias_path = optarg; remove_trailing_slash((char *)bias_path); break;
            case 'd': dark_path = optarg; remove_trailing_slash((char *)dark_path); break;
            case 'f': flat_path = optarg; remove_trailing_slash((char *)flat_path); break;
            case 'l': in_path = optarg; remove_trailing_slash((char *)in_path); break;
            case 'o': out_path = optarg; break;
            default:
                fprintf(stderr, "Usage: %s --light <input/dir> --bias <bias/dir> --dark <dark/dir> --flat <flat/dir> [--output <output/dir>]\n", argv[0]);
                return 1;
        }
    }

    if (!in_path || !bias_path || !dark_path || !flat_path) {
        fprintf(stderr, "Light, bias, dark and flat directories/files are required.\n");
        return 1;
    }

    // Inizializza CUDA
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    CHECK(cudaSetDevice(dev));
    PrefetchDeviceArg devLoc = make_prefetch_device_arg(dev);


    // ==================================================================
    // 1. Calibration

    // ===== 1.1 Bias =====
    
    long bias_width=0, bias_height=0, bias_n_chan=0, npixels=0;
    int bias_count=0;
    float *master_bias = nullptr;

    // First, check if there's already a master_bias in the output directory
    string master_bias_file;
    if (find_latest_master_file(out_path, "bias", master_bias_file)) {
        // Load existing master bias
        fitsfile *fpbias = nullptr;
        open_fits(master_bias_file.c_str(), &fpbias);
        get_fits_dimensions(fpbias, &bias_width, &bias_height, &bias_n_chan);
        if (bias_width <= 0 || bias_height <= 0 || bias_n_chan != 1) {
            fprintf(stderr, "Invalid master bias dimensions\n");
            return 1;
        }
        npixels = bias_width * bias_height;
        CHECK(cudaMallocManaged(&master_bias, npixels * sizeof(float)));
        get_fits_data(fpbias, npixels, master_bias);
    }
    // If no existing master, check if bias_path is a file (direct master bias)
    else if (is_regular_file(bias_path)) {
        fitsfile *fpbias = nullptr;
        open_fits(bias_path, &fpbias);
        get_fits_dimensions(fpbias, &bias_width, &bias_height, &bias_n_chan);
        if (bias_width <= 0 || bias_height <= 0 || bias_n_chan != 1) {
            fprintf(stderr, "Invalid master bias dimensions\n");
            return 1;
        }
        npixels = bias_width * bias_height;
        CHECK(cudaMallocManaged(&master_bias, npixels * sizeof(float)));
        get_fits_data(fpbias, npixels, master_bias);  // automatically close the fits file
    }
    // Otherwise, compute from bias directory
    else if (is_directory(bias_path)) {
        if (check_directory(bias_path, &bias_count, &bias_width, &bias_height, &bias_n_chan, 1) != 0) {
            fprintf(stderr, "Error checking bias directory\n");
            return 1;
        }

        npixels = bias_width * bias_height;

        // bias images memory allocation
        uint16_t *bias_all = nullptr;
        CHECK(cudaMallocManaged(&bias_all, npixels * bias_count * sizeof(uint16_t)));

        // read all bias images
        if (load_images_to_memory_prefetch(bias_path, bias_all, bias_width, bias_height, bias_n_chan, bias_count, dev) != 0) {
            fprintf(stderr, "Error loading bias images\n");
            return 1;
        }

        // master bias memory allocation
        CHECK(cudaMallocManaged(&master_bias, npixels * sizeof(float)));
        CHECK(cudaMemPrefetchAsync(master_bias, npixels * sizeof(float), devLoc, 0));

        // master bias computation
        masterBias_gpu(bias_all, master_bias, bias_width, bias_height, bias_count);

        // free bias images memory
        CHECK(cudaFree(bias_all));

        save_image_fits(out_path, "master_bias", master_bias, bias_width, bias_height, 1);
    } else {
        printf("Error: bias path '%s' is nor a file or a directory\n", bias_path);
        return 1;
    }


    // ===== 1.2 Dark =====

    long dark_width=0, dark_height=0, dark_n_chan=0;
    int dark_count=0;
    float *master_dark = nullptr;

    // check if there's already a master_dark in the output directory
    string master_dark_file;
    if (find_latest_master_file(out_path, "dark", master_dark_file)) {
        // Load existing master dark
        fitsfile *fpdark = nullptr;
        open_fits(master_dark_file.c_str(), &fpdark);
        get_fits_dimensions(fpdark, &dark_width, &dark_height, &dark_n_chan);
        if (dark_width != bias_width || dark_height != bias_height || dark_n_chan != 1) {
            fprintf(stderr, "Invalid master dark dimensions - they need to be the same as master bias!\n");
            return 1;
        }
        CHECK(cudaMallocManaged(&master_dark, npixels * sizeof(float)));
        get_fits_data(fpdark, npixels, master_dark);
    }
    // If no existing master, check if dark_path is a file (direct master dark)
    else if (is_regular_file(dark_path)) {
        fitsfile *fpdark = nullptr;
        open_fits(dark_path, &fpdark);
        get_fits_dimensions(fpdark, &dark_width, &dark_height, &dark_n_chan);
        if (dark_width != bias_width || dark_height != bias_height || dark_n_chan != 1) {
            fprintf(stderr, "Invalid master dark dimensions - they need to be the same as master bias!\n");
            return 1;
        }
        CHECK(cudaMallocManaged(&master_dark, npixels * sizeof(float)));
        get_fits_data(fpdark, npixels, master_dark);
    }
    // Otherwise, compute from dark directory
    else if (is_directory(dark_path)) {
        if (check_directory(dark_path, &dark_count, &dark_width, &dark_height, &dark_n_chan, 1) != 0) {
            fprintf(stderr, "Error checking dark directory\n");
            return 1;
        }
        if (dark_width != bias_width || dark_height != bias_height || dark_n_chan != 1) {
            fprintf(stderr, "Invalid master dark dimensions - they need to be the same as master bias!\n");
            return 1;
        }

        // dark images memory allocation
        uint16_t *dark_all = nullptr;
        CHECK(cudaMallocManaged(&dark_all, npixels * dark_count * sizeof(uint16_t)));

        // read all dark images
        if (load_images_to_memory_prefetch(dark_path, dark_all, dark_width, dark_height, dark_n_chan, dark_count, dev) != 0) {
            fprintf(stderr, "Error loading dark images\n");
            return 1;
        }

        // master dark memory allocation
        CHECK(cudaMallocManaged(&master_dark, npixels * sizeof(float)));
        CHECK(cudaMemPrefetchAsync(master_dark, npixels * sizeof(float), devLoc, 0));

        // master dark computation
        masterDark_gpu(dark_all, master_bias, master_dark, dark_width, dark_height, dark_count);

        // free dark images memory
        CHECK(cudaFree(dark_all));

        // save to output
        save_image_fits(out_path, "master_dark", master_dark, dark_width, dark_height, 1);
    } else {
        printf("Error: dark path '%s' is nor a file or a directory\n", dark_path);
        return 1;
    }


    // ===== 1.3 flat =====

    long flat_width=0, flat_height=0, flat_n_chan=0;
    int flat_count=0;
    float *master_flat = nullptr;

    // First, check if there's already a master_flat in the output directory
    string master_flat_file;
    if (find_latest_master_file(out_path, "flat", master_flat_file)) {
        // Load existing master flat
        fitsfile *fpflat = nullptr;
        open_fits(master_flat_file.c_str(), &fpflat);
        get_fits_dimensions(fpflat, &flat_width, &flat_height, &flat_n_chan);
        if (flat_width != bias_width || flat_height != bias_height || flat_n_chan != 1) {
            fprintf(stderr, "Invalid master flat dimensions - they need to be the same as master bias!\n");
            return 1;
        }
        CHECK(cudaMallocManaged(&master_flat, npixels * sizeof(float)));
        get_fits_data(fpflat, npixels, master_flat);
    }
    // If no existing master, check if flat_path is a file (direct master flat)
    else if (is_regular_file(flat_path)) {
        fitsfile *fpflat = nullptr;
        open_fits(flat_path, &fpflat);
        get_fits_dimensions(fpflat, &flat_width, &flat_height, &flat_n_chan);
        if (flat_width != bias_width || flat_height != bias_height || flat_n_chan != 1) {
            fprintf(stderr, "Invalid master flat dimensions - they need to be the same as master bias!\n");
            return 1;
        }
        CHECK(cudaMallocManaged(&master_flat, npixels * sizeof(float)));
        get_fits_data(fpflat, npixels, master_flat);
    }
    // Otherwise, compute from flat directory
    else if (is_directory(flat_path)) {
        if (check_directory(flat_path, &flat_count, &flat_width, &flat_height, &flat_n_chan, 1) != 0) {
            fprintf(stderr, "Error checking flat directory\n");
            return 1;
        }
        if (flat_width != bias_width || flat_height != bias_height || flat_n_chan != 1) {
            fprintf(stderr, "Invalid master flat dimensions - they need to be the same as master bias!\n");
            return 1;
        }

        // flat images memory allocation
        uint16_t *flat_all = nullptr;
        CHECK(cudaMallocManaged(&flat_all, npixels * flat_count * sizeof(uint16_t)));

        // read all flat images
        if (load_images_to_memory_prefetch(flat_path, flat_all, flat_width, flat_height, flat_n_chan, flat_count, dev) != 0) {
            fprintf(stderr, "Error loading flat images\n");
            return 1;
        }

        // master flat memory allocation
        CHECK(cudaMallocManaged(&master_flat, npixels * sizeof(float)));
        CHECK(cudaMemPrefetchAsync(master_flat, npixels * sizeof(float), devLoc, 0));

        // master flat computation
        masterFlat_gpu(flat_all, master_bias, master_flat, flat_width, flat_height, flat_count);

        // free flat images memory
        CHECK(cudaFree(flat_all));

        // save to output
        save_image_fits(out_path, "master_flat", master_flat, flat_width, flat_height, 1);
    } else {
        printf("Error: flat path '%s' is nor a file or a directory\n", flat_path);
        return 1;
    }

// ==================================================================
// 2. Light calibration &Memory allocation (light -> calibrated -> debayering -> alignment )

    // maybe I can reuse some memory???

    // light        (n*1)  -|
    // calibrated   (n*1)  -|--> (n*3) will reuse for alignment result
    // free buffer  (n*1)  -|
    // debayering   (n*3)
    // alignment    (n*3)

    // so I need to allocate light_count*npixels*6*sizeof(u_int16_t) memory
    // and I will map
    // light to         from (start)                         to (start + npixels*light_count - 1)
    // calibrated to    from (start + npixels*light_count)   to (start + 2*npixels*light_count - 1)
    // free buffer      from (start + 2*npixels*light_count) to (start + 3*npixels*light_count - 1)
    // debayering       from (start + 3*npixels*light_count) to (start + 6*npixels*lignt_count - 1)

    // alignment        from (start)                         to (start + 3*npixels*light_count - 1)


    // check directory
    long light_width=0, light_height=0, light_n_chan=0;
    int light_count=0;
    if (check_directory(in_path, &light_count, &light_width, &light_height, &light_n_chan, 1) != 0) {
        fprintf(stderr, "Error checking light directory\n");
        return 1;
    }
    if (light_width != bias_width || light_height != bias_height || light_n_chan != bias_n_chan) {
        fprintf(stderr, "Invalid light dimensions - they need to be the same as master bias!\n");
        return 1;
    }

    // memory allocation for images
    uint16_t *light_all = nullptr, *calibrated_all = nullptr;
    CHECK(cudaMallocManaged(&light_all, npixels * light_count * sizeof(uint16_t)));
    CHECK(cudaMallocManaged(&calibrated_all, npixels * light_count * sizeof(uint16_t)));
    CHECK(cudaMemPrefetchAsync(calibrated_all, npixels * light_count * sizeof(uint16_t), devLoc, 0));
    
    
    printf("memory allocation done, used %f Mb\n", 2*npixels*light_count*sizeof(uint16_t) / 1e6);

    // memory allocation for timestamps of images
    double *timestamps = new double[light_count];

    // reading all light images
    if (load_images_to_memory_prefetch(in_path, light_all, light_width, light_height, light_n_chan, light_count, dev, timestamps) != 0) {
        fprintf(stderr, "Error loading light images\n");
        return 1;
    }
    printf("loaded %d light images\n", light_count);
    int central_image_index = find_mid_image_index(timestamps, light_count);
    printf("Reference image index:%d\n", central_image_index);

    // calibration
    calibrateLights_gpu(light_all, master_bias, master_dark, master_flat, calibrated_all, light_width, light_height, light_count);
    printf("calibration done\n");
    
    // free light_all memory and masters memory
    CHECK(cudaFree(light_all));
    CHECK(cudaFree(master_bias));
    CHECK(cudaFree(master_dark));
    CHECK(cudaFree(master_flat));


    // ==================================================================
    // 3. Debayering

    uint16_t *debayered_all = nullptr, *aligned_all = nullptr;
    uint16_t *mem_block = nullptr;
    CHECK(cudaMallocManaged(&mem_block, npixels*3*(light_count+1)*sizeof(uint16_t)));
    CHECK(cudaMemPrefetchAsync(mem_block, npixels*3*(light_count+1)*sizeof(uint16_t), devLoc, 0));

    debayered_all = mem_block + npixels*3;
    aligned_all   = mem_block;
    
    demosaic_mhc_rggb_gpu(calibrated_all, debayered_all, light_width, light_height, light_count);
    printf("debayering done\n");

    // free calibrated_all memory
    CHECK(cudaFree(calibrated_all));

    // ==================================================================
    // 4. Alignment


    // ===== 4.1 allocate memory =====
    uint16_t *gray_img = nullptr;
    uint8_t *threshold_img = nullptr;
    star *stars = nullptr;
    uint32_t *num_stars = nullptr;
    CHECK(cudaMallocManaged(&gray_img, npixels * sizeof(uint16_t)));
    CHECK(cudaMallocManaged(&threshold_img, npixels * sizeof(uint8_t)));
    CHECK(cudaMallocManaged(&stars, 1024 * sizeof(star)));               // TODO: parametrize max starts
    CHECK(cudaMallocManaged(&num_stars, sizeof(uint32_t)));
    CHECK(cudaMemPrefetchAsync(gray_img, npixels*sizeof(uint16_t), devLoc, 0));
    CHECK(cudaMemPrefetchAsync(threshold_img, npixels*sizeof(uint8_t), devLoc, 0));

    const int MAX_STARS_TO_DETECT = 500;  // Limit stars for better matching

    

    // ===== 4.2 grayscale =====
    uint16_t *central_image = debayered_all + (central_image_index * npixels * 3);
    to_grayscale_planar_gpu(central_image, gray_img, npixels);
    //save_image_fits(out_path, "central_image_calibrated", central_image, light_width, light_height, 3);


    // ===== 4.3 thresholding =====
    threshold_params t_par;
        t_par.window_size = 201;    
        t_par.threshold_scale = 0.7F;
    compute_threshold_gpu(gray_img, threshold_img, light_width, light_height, t_par);
    printf("threshold done\n");

    // ===== 4.4 detect stars =====
    detect_stars_gpu(threshold_img, light_width, light_height, 100, 10, stars, num_stars, MAX_STARS_TO_DETECT);

    // ===== 4.5 populate star details =====
    star_detail *stars_details = nullptr;
    CHECK(cudaMallocManaged(&stars_details, *num_stars * sizeof(star_detail)));
    populate_star_details_gpu(stars_details, stars, *num_stars, central_image, gray_img, light_width, npixels);

    // ===== 4.6 keypoints and descriptors ===== (for opencv)
    std::vector<cv::KeyPoint> keypoints_central_img;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors_central_img, descriptors;

    bool ok = build_star_descriptors(
        stars_details, *num_stars, light_width, light_height, keypoints_central_img, descriptors_central_img);
    CHECK(cudaFree(stars_details));
    if (!ok) {
        printf("Not enough stars to build descriptors");
        return 1;
    }

    // ===== 4.7.0 create aligned output directory =====
    string aligned_dir = string(out_path) + "/aligned";
    mkdir(aligned_dir.c_str(), 0755);

    // ===== 4.7 for every image =====
    int index = 0;
    for (int i = 0; i < light_count; i++) {
        uint16_t *current_img = debayered_all  + (i     * npixels * 3);
        uint16_t *dest        = aligned_all    + (index * npixels * 3);

        if (i == central_image_index) {
            // Copia diretta dell'immagine centrale (senza warp)
            CHECK(cudaMemcpy(dest, current_img, npixels * 3 * sizeof(uint16_t), cudaMemcpyDefault));
            char aligned_name[64];
            snprintf(aligned_name, sizeof(aligned_name), "aligned_%04d", i);
            //save_image_fits(aligned_dir, aligned_name, dest, light_width, light_height, 3);
            index++;
            continue;
        }

        to_grayscale_planar_gpu(current_img, gray_img, npixels);
        compute_threshold_gpu(gray_img, threshold_img, light_width, light_height, t_par);

        detect_stars_gpu(threshold_img, light_width, light_height, 100, 10, stars, num_stars, MAX_STARS_TO_DETECT);

        star_detail *cur_stars_details = nullptr;
        CHECK(cudaMallocManaged(&cur_stars_details, *num_stars * sizeof(star_detail)));
        populate_star_details_gpu(cur_stars_details, stars, *num_stars, current_img, gray_img, light_width, npixels);

        ok = build_star_descriptors(
            cur_stars_details, *num_stars, light_width, light_height, keypoints, descriptors);
        CHECK(cudaFree(cur_stars_details));

        if (!ok) {
            printf("Not enough stars to build descriptors for image %d, skipping\n", i);
            continue;
        }

        float ratio_threshold = 0.7F;
        cv::Mat affine_2x3 = estimate_affine_partial_stars(
            keypoints_central_img, descriptors_central_img,
            keypoints, descriptors, ratio_threshold);

        warp_affine_planar_gpu(current_img, dest, affine_2x3, light_width, light_height);

        // Save aligned image
        char aligned_name[64];
        snprintf(aligned_name, sizeof(aligned_name), "aligned_%04d", i);
        //save_image_fits(aligned_dir, aligned_name, dest, light_width, light_height, 3);
        index++;
    }

    int aligned_count = index;
    printf("Aligned %d images\n", aligned_count);

    // ===== 4.8 cleanup =====
    CHECK(cudaFree(gray_img));
    CHECK(cudaFree(threshold_img));
    CHECK(cudaFree(stars));
    CHECK(cudaFree(num_stars));

    delete[] timestamps;


    // ==================================================================
    // 5. stacking
    
    // ===== 5.1 stacked image memory allocation =====
    uint16_t *stacked_img = nullptr;
    CHECK(cudaMallocManaged(&stacked_img, npixels * 3 * sizeof(uint16_t)));

    // ===== 5.2 stacking =====
    winsorized_sigma_clipping_gpu(aligned_all, stacked_img, (uint16_t)aligned_count, npixels*3);

    // ===== 5.3 saving result =====
    save_image_fits(out_path, "stacked", stacked_img, light_width, light_height, 3);

    // ===== 5.4 cleanup =====
    CHECK(cudaFree(stacked_img));
    CHECK(cudaFree(mem_block));

    return 0;
}

