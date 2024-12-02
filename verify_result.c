#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fitsio.h>

/* compilation command:
gcc -o verify_result verify_result.c -lcfitsio -O3
*/


void open_fits(char *file_path, fitsfile **fptr) {
    int status = 0;
    if (fits_open_file(fptr, file_path, READONLY, &status)) {
        fits_report_error(stderr, status);
        fprintf(stderr, "Could not open input file\n");
        exit(1);
    }
}

void get_fits_data(fitsfile *fptr, size_t npixels, u_int16_t *fits_data) {
    int status = 0;
    if (fits_read_img(fptr, TUSHORT, 1, npixels, NULL, fits_data, NULL, &status)) {
        fits_report_error(stderr, status);
        fprintf(stderr, "Could not read input file\n");
        fits_close_file(fptr, &status);
        free(fits_data);
        exit(1);
    }
}

void get_image_dimensions(fitsfile *fptr, int *width, int *height, int *depth) {
    int status = 0;
    int naxis;
    long naxes[3] = {1, 1, 1};
    if (fits_get_img_dim(fptr, &naxis, &status) || fits_get_img_size(fptr, 3, naxes, &status)) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        exit(1);
    }
    *width = naxes[0];
    *height = naxes[1];
    if (naxis == 2) {
        *depth = 1;
        return;
    }
    *depth = naxes[2];
}


int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <file1> <file2>\n", argv[0]);
        return 1;
    }

    fitsfile *fptr1, *fptr2;
    int status = 0;
    open_fits(argv[1], &fptr1);
    open_fits(argv[2], &fptr2);

    int width1, height1, depth1;
    int width2, height2, depth2;

    get_image_dimensions(fptr1, &width1, &height1, &depth1);
    get_image_dimensions(fptr2, &width2, &height2, &depth2);

    if (width1 != width2 || height1 != height2 || depth1 != depth2) {
        printf("Files are different dimensions\n");
        fits_close_file(fptr1, &status);
        fits_close_file(fptr2, &status);
        return 0;
    }

    long npixels = width1 * height1 * depth1;

    u_int16_t *fits_data1 = (u_int16_t *)malloc(npixels * sizeof(u_int16_t));
    u_int16_t *fits_data2 = (u_int16_t *)malloc(npixels * sizeof(u_int16_t));

    get_fits_data(fptr1, npixels, fits_data1);
    get_fits_data(fptr2, npixels, fits_data2);
/*
    if (memcmp(fits_data1, fits_data2, npixels * sizeof(u_int16_t)) == 0) {
        printf("Files are identical\n");
    } else {
        printf("Files are different\n");
    }*/
    int identical = 1;
    for (long i = 0; i < npixels; i++) {
        if (fits_data1[i] != fits_data2[i]) {
            identical = 0;
            printf("Difference at pixel %ld: file1 = %u, file2 = %u\n", i, fits_data1[i], fits_data2[i]);
        }
    }

    if (identical) {
        printf("Files are identical\n");
    } else {
        printf("Files are different\n");
    }

    free(fits_data1);
    free(fits_data2);
    fits_close_file(fptr1, &status);
    fits_close_file(fptr2, &status);

    return 0;
}