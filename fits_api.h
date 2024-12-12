#ifndef FITS_API_H
#define FITS_API_H

#include <stdio.h>
#include <fitsio.h>
#include <dirent.h>
#include <string.h>
#include <ctime>


void remove_trailing_slash(char *in_dir) {
    if (in_dir[strlen(in_dir) - 1] == '/') {
        in_dir[strlen(in_dir) - 1] = '\0';
    }
}


// funzioni per file FITS
void open_fits(char *file_path, fitsfile **fptr) {
    int status = 0;
    if (fits_open_file(fptr, file_path, READONLY, &status)) {
        fits_report_error(stderr, status);
        fprintf(stderr, "Could not open input file\n");
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

void print_fits_metadata(fitsfile *fptr) {
    int nkeys, status = 0;
    char card[FLEN_CARD];
    if (fits_get_hdrspace(fptr, &nkeys, NULL, &status)) {
        fits_report_error(stderr, status);
        fprintf(stderr, "Could not get header space\n");
        exit(1);
    }

    printf("Header information:\n");
    for (int i = 1; i <= nkeys; i++) {
        if (fits_read_record(fptr, i, card, &status)) {
            fits_report_error(stderr, status);
            fprintf(stderr, "Could not read header record\n");
            exit(1);
        }
        printf("%s\n", card);
    }
}

void save_image_fits(char const *output_dir_path, u_int16_t *image_data, int width, int height, int depth) {
    fitsfile *fptr;
    int status = 0;

    char output_path[1024];
    strcpy(output_path, output_dir_path);
    remove_trailing_slash(output_path);

    //aggiungo data, ora ed estensione al nome del file
    char timestamp_str[29];
    time_t rawtime;
    struct tm *timeinfo;
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(timestamp_str, sizeof(timestamp_str), "/image_%Y%m%d_%H%M%S.fits", timeinfo);
    strcat(output_path, timestamp_str);

    if (fits_create_file(&fptr, output_path, &status)) {
        if (status == FILE_NOT_CREATED) {
            status = 0; // Reset status before attempting to overwrite
            if (fits_open_file(&fptr, output_path, READWRITE, &status)) {
                fits_report_error(stderr, status);
                exit(1);
            }
            if (fits_delete_file(fptr, &status)) {
                fits_report_error(stderr, status);
                exit(1);
            }
            status = 0; // Reset status before creating the file again
            if (fits_create_file(&fptr, output_path, &status)) {
                fits_report_error(stderr, status);
                exit(1);
            }
        } else {
            fits_report_error(stderr, status);
            exit(1);
        }
    }

    if (depth == 1) {
        long naxes[2] = {width, height};
        if (fits_create_img(fptr, USHORT_IMG, 2, naxes, &status)) {
            fits_report_error(stderr, status);
            exit(1);
        }
    } else {
        long naxes[3] = {width, height, depth};
        if (fits_create_img(fptr, USHORT_IMG, 3, naxes, &status)) {
            fits_report_error(stderr, status);
            exit(1);
        }
    }

    // Write image data
    if (fits_write_img(fptr, TUSHORT, 1, width * height * depth, image_data, &status)) {
        fits_report_error(stderr, status);
        exit(1);
    }

    printf("Image saved to %s, metadata:\n", output_path);
    print_fits_metadata(fptr);
    fits_close_file(fptr, &status);
}

#endif // FITS_API_H
