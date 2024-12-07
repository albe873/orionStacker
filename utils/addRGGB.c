#include <stdio.h>
#include <fitsio.h>

/*
gcc addRGGB.c -o addRGGB -lcfitsio
*/

void open_fits(char *file_path, fitsfile **fptr) {
    int status = 0;
    if (fits_open_file(fptr, file_path, READONLY, &status)) {
        fits_report_error(stderr, status);
        fprintf(stderr, "Could not open input file\n");
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

void get_image_dimensions(fitsfile *fptr, int *width, int *height) {
    int status = 0;
    int naxis;
    long naxes[2] = {1, 1};
    if (fits_get_img_dim(fptr, &naxis, &status) || fits_get_img_size(fptr, 2, naxes, &status)) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        exit(1);
    }
    *width = naxes[0];
    *height = naxes[1];
}


void get_fits_data(fitsfile *fptr, size_t npixels, float *fits_data) {
    int status = 0;
    if (fits_read_img(fptr, TFLOAT, 1, npixels, NULL, fits_data, NULL, &status)) {
        fits_report_error(stderr, status);
        fprintf(stderr, "Could not read input file\n");
        fits_close_file(fptr, &status);
        free(fits_data);
        exit(1);
    }
}

void get_fits_data_u_int16(fitsfile *fptr, size_t npixels, u_int16_t *fits_data) {
    int status = 0;
    if (fits_read_img(fptr, TUSHORT, 1, npixels, NULL, fits_data, NULL, &status)) {
        fits_report_error(stderr, status);
        fprintf(stderr, "Could not read input file\n");
        fits_close_file(fptr, &status);
        free(fits_data);
        exit(1);
    }
}


void save_image_fits_u_int16(char const *output_path, u_int16_t *image_data, int width, int height) {
    fitsfile *fptr;
    int status = 0;

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

    if (fits_create_img(fptr, USHORT_IMG, 2, (long[]){width, height}, &status)) {
        fits_report_error(stderr, status);
        exit(1);
    }

    // Write image data
    if (fits_write_img(fptr, TUSHORT, 1, width * height, image_data, &status)) {
        fits_report_error(stderr, status);
        exit(1);
    }

    // Add Bayer filter information
    char *bayer_pattern = "RGGB"; // Example Bayer pattern
    if (fits_update_key(fptr, TSTRING, "BAYERPAT", bayer_pattern, "Bayer color pattern", &status)) {
        fits_report_error(stderr, status);
        exit(1);
    }

    printf("Image saved to %s, metadata:\n", output_path);
    print_fits_metadata(fptr);
    fits_close_file(fptr, &status);
}

void save_image_fits(char const *output_path, float *image_data, int width, int height) {
    fitsfile *fptr;
    int status = 0;

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

    if (fits_create_img(fptr, FLOAT_IMG, 2, (long[]){width, height}, &status)) {
        fits_report_error(stderr, status);
        exit(1);
    }

    // Write image data
    if (fits_write_img(fptr, TFLOAT, 1, width * height, image_data, &status)) {
        fits_report_error(stderr, status);
        exit(1);
    }

    // Add Bayer filter information
    char *bayer_pattern = "RGGB"; // Example Bayer pattern
    if (fits_update_key(fptr, TSTRING, "BAYERPAT", bayer_pattern, "Bayer color pattern", &status)) {
        fits_report_error(stderr, status);
        exit(1);
    }

    printf("Image saved to %s, metadata:\n", output_path);
    print_fits_metadata(fptr);
    fits_close_file(fptr, &status);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <file_path>\n", argv[0]);
        return 1;
    }

    fitsfile *fptr;

    open_fits(argv[1], &fptr);
    print_fits_metadata(fptr);
    char output_path[] = "output.fits";

    int width, height, npixels, status = 0;
    get_image_dimensions(fptr, &width, &height);
    npixels = width * height;

    float *fits_data_h;
    u_int16_t *fits_data_u_int16 = (u_int16_t *) malloc(npixels * sizeof(u_int16_t));
    fits_data_h = (float *) malloc(npixels * sizeof(float));
    get_fits_data_u_int16(fptr, npixels, fits_data_u_int16);
    //get_fits_data(fptr, npixels, fits_data_h);
    fits_close_file(fptr, &status);

    save_image_fits_u_int16(output_path, fits_data_u_int16, width, height);

    exit(0);
}