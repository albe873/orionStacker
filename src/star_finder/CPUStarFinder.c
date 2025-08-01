#include "../common/fits_api.h"
#include "host_starFinder.h"

#include <stdio.h>
#include <getopt.h>
#include <unistd.h>

/*  compile

gcc CPUStarFinder.c -o CPUStarFinder -lcfitsio -lm -O3 -march=native -Wall
*/

double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}


int main(int argc, char **argv) {

    char *filename = NULL;
    int opt, option_index = 0;
    u_int16_t threshold = 1000;
    u_int8_t reduce_factor = 8;
    u_int16_t window_size = 255;
    u_int16_t max_star_size = 75;

    enum ThresholdType {
        TR_SIMPLE,
        TR_ADAPTIVE,
        TR_FAST_ADAPTIVE
    };
    int threshold_algorithm = TR_SIMPLE;

    static struct option long_options[] = {
        {"input-file", required_argument, 0, 'f'},
        {"threshold", optional_argument, 0, 't'},
        {"reduce-factor", optional_argument, 0, 'r'},
        {"threshold-algorith", optional_argument, 0, 'a'},
        {"window-size", optional_argument, 0, 'w'},
        {"max-star-size", optional_argument, 0, 'm'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "f:t:r:a:w:m:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'f':
                filename = optarg;
                break;
            case 't':
                threshold = atoi(optarg);
                break;
            case 'r':
                reduce_factor = atoi(optarg);
                break;
            case 'a':
                if (strcmp(optarg, "simple") == 0) {
                    threshold_algorithm = TR_SIMPLE;
                } else if (strcmp(optarg, "adaptive") == 0) {
                    threshold_algorithm = TR_ADAPTIVE;
                } else if (strcmp(optarg, "fast-adaptive") == 0) {
                    threshold_algorithm = TR_FAST_ADAPTIVE;
                } else {
                    fprintf(stderr, "Invalid threshold algorithm, using default\n");
                }
                break;
            case 'w':
                window_size = atoi(optarg);
                break;
            case 'm':
                max_star_size = atoi(optarg);
                break;
            default:
                fprintf(stderr, "Usage: %s --input-file <image.fits>\n", argv[0]);
                return 1;
        }
    }

    if (filename == NULL) {
        fprintf(stderr, "Usage: %s --input-file <image.fits>\n", argv[0]);
        return 1;
    }

    // Apre il file FITS
    fitsfile *fptr = NULL;
    int status = 0;
    long width, height, depth;
    open_fits(filename, &fptr);
    get_image_dimensions(fptr, &width, &height, &depth);
    u_int64_t totpixels = width * height * depth;
    u_int64_t npixels = width * height;

    u_int16_t *fits_data = NULL;
    fits_data = (u_int16_t *) malloc(totpixels * sizeof(u_int16_t));
 

    u_int16_t *gray_image = NULL;
    gray_image = (u_int16_t *) malloc(npixels * sizeof(u_int16_t));

    u_int16_t *reduced_image = NULL;
    reduced_image = (u_int16_t *) malloc((npixels / reduce_factor / reduce_factor) * sizeof(u_int16_t));
    
    u_int16_t *threshold_image = NULL;
    threshold_image = (u_int16_t *) malloc(npixels * sizeof(u_int16_t));

    get_fits_data(fptr, totpixels, fits_data);
    fits_close_file(fptr, &status);

    // se depth == 1, allora bisogna applicare il filtro di bayer???
    double t_start, t_elapsed;
    t_start = cpuSecond();

    to_grayscale_fits_cpu(fits_data, gray_image, npixels);

    switch (threshold_algorithm) {
        case TR_SIMPLE:
            simple_threshold_cpu(gray_image, threshold_image, npixels, threshold);
            break;
        case TR_ADAPTIVE:
            adaptiveThresholding_cpu(gray_image, threshold_image, width, height, window_size, threshold);
            break;
        case TR_FAST_ADAPTIVE:
            reduce_image_cpu(gray_image, reduced_image, width, height, reduce_factor);
            adaptiveThresholdingApprossimative_cpu(gray_image, threshold_image, width, height, 
                                                 reduced_image, reduce_factor, window_size, threshold);
            break;
    }

    detect_stars_cpu(threshold_image, fits_data, width, height, max_star_size);

    t_elapsed = cpuSecond() - t_start;
    printf("Elapsed time: %f\n", t_elapsed);

    save_image_fits(".", "threshold", threshold_image, width, height, 1);
    save_image_fits(".", "detect_output", fits_data, width, height, 3);
}
