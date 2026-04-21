#include <stdio.h>
#include <fitsio.h>
#include <string.h>
#include <time.h>
#include <string>
#include "include/common.h"
#include "opencv2/imgcodecs.hpp"

using namespace std;
using namespace cv;

void open_fits(string file_path, fitsfile **fptr) {
    int status = 0;
    if (fits_open_file(fptr, file_path.c_str(), READONLY, &status)) {
        fits_report_error(stderr, status);
        fprintf(stderr, "Could not open input file\n");
        exit(1);
    }
}

void get_fits_dimensions(fitsfile *fptr, long *width, long *height, long *n_chan) {
    int status = 0;
    int naxis;
    long naxes[3] = {1, 1, 1};
    if (fits_get_img_dim(fptr, &naxis, &status) || fits_get_img_size(fptr, 3, naxes, &status)) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        exit(1);
    }
    if (naxis < 2 || naxis > 3) {
        fprintf(stderr, "Only 2D images are supported\n");
        fits_close_file(fptr, &status);
        exit(1);
    }
    if (naxes[0] < 1 || naxes[1] < 1) {
        fprintf(stderr, "Invalid image dimensions\n");
        fits_close_file(fptr, &status);
        exit(1);
    }
    *width = naxes[0];
    *height = naxes[1];
    if (naxis == 3) {
        if (naxes[2] < 1 || naxes[2] > 3) {
            fprintf(stderr, "Invalid number of channels\n");
            fits_close_file(fptr, &status);
            exit(1);
        }
        *n_chan = naxes[2];
    }
    else {
        *n_chan = 1;
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

inline void check_str_length(string &str, size_t max_length, const char *error_message) {
    if (str.length() > max_length) {
        fprintf(stderr, "%s\n", error_message);
        exit(1);
    }
}

inline void add_timestamp(string &str) {
    char timestamp_str[17];
    time_t rawtime;
    struct tm *timeinfo;
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(timestamp_str, sizeof(timestamp_str), "_%Y%m%d_%H%M%S", timeinfo);
    str += timestamp_str;
}

Mat to_bgr_mat_from_planar_data(u_int16_t *image_data, long width, long height) {
    size_t planeSize = (size_t)width * (size_t)height;
    auto *r = image_data;
    auto *g = image_data + planeSize;
    auto *b = image_data + 2 * planeSize;
    cv::Mat img;
    cv::merge(std::vector<cv::Mat>{
        cv::Mat((int)height, (int)width, CV_16UC1, (void*) b),
        cv::Mat((int)height, (int)width, CV_16UC1, (void*) g),
        cv::Mat((int)height, (int)width, CV_16UC1, (void*) r)
    }, img);
    return img;
}


void save_image_fits(string output_dir_path, string file_name, u_int16_t *image_data, long width, long height, long n_chan) {

    check_str_length(output_dir_path, 255, "Output directory name too long");
    check_str_length(file_name, 230, "File name too long");

    if (width < 1 || height < 1 || (n_chan != 1 && n_chan != 3)) {
        fprintf(stderr, "Invalid image dimensions or number of channels\n");
        return;
    }

    string output_path = output_dir_path;
    if (output_path.back() != '/' && !output_path.empty())
        output_path += "/";
    output_path += file_name;

    add_timestamp(output_path);

    // add .fits extension if not present
    if (output_path.find(".fits") == std::string::npos && output_path.find(".FITS") == std::string::npos) {
        output_path += ".fits";
    }

    fitsfile *fptr;
    int status = 0;
    if (fits_create_file(&fptr, output_path.c_str(), &status)) {
        if (status == FILE_NOT_CREATED) {
            // File already exists, try to delete and recreate
            status = 0;
            if (fits_open_file(&fptr, output_path.c_str(), READWRITE, &status)) {
                fits_report_error(stderr, status);
                return;
            }
            if (fits_delete_file(fptr, &status)) {
                fits_report_error(stderr, status);
                return;
            }
            status = 0;
            if (fits_create_file(&fptr, output_path.c_str(), &status)) {
                fits_report_error(stderr, status);
                return;
            }
        } else {
            fits_report_error(stderr, status);
            return;
        }
    }

    if (n_chan == 1) {
        long naxes[2] = {width, height};
        if (fits_create_img(fptr, USHORT_IMG, 2, naxes, &status)) {
            fits_report_error(stderr, status);
            return;
        }
    } else {
        long naxes[3] = {width, height, n_chan};
        if (fits_create_img(fptr, USHORT_IMG, 3, naxes, &status)) {
            fits_report_error(stderr, status);
            return;
        }
    }

    // Write image data
    if (fits_write_img(fptr, TUSHORT, 1, width * height * n_chan, image_data, &status)) {
        fits_report_error(stderr, status);
        return;
    }

    printf("Image saved to %s, metadata:\n", output_path.c_str());
    print_fits_metadata(fptr);
    fits_close_file(fptr, &status);
}



void save_image_tiff(string output_dir_path, string file_name, u_int16_t *image_data, long width, long height, long n_chan) {
    // Check string lengths
    check_str_length(output_dir_path, 255, "Output directory name too long");
    check_str_length(file_name, 230, "File name too long");

    // Construct output path
    string output_path = output_dir_path;
    if (output_path.back() != '/' && !output_path.empty())
        output_path += "/";
    output_path += file_name;

    // Add timestamp
    //add_timestamp(output_path);
    
    // add .tiff extension if not present
    if (output_path.find(".tiff") == std::string::npos && output_path.find(".tif") == std::string::npos) {
        output_path += ".tiff";
    }

    // Create mat object    
    cv::Mat img;
    if (n_chan == 3) {
        img = to_bgr_mat_from_planar_data(image_data, width, height);
    } else if (n_chan == 1) {
        img = cv::Mat((int)height, (int)width, CV_16UC1, (void*)image_data);
    } else {
        fprintf(stderr, "Unsupported number of channels: %ld\n", n_chan);
        return;
    }

    // Write TIFF
    if (!cv::imwrite(output_path, img)) {
        fprintf(stderr, "Failed to write TIFF %s\n", output_path.c_str());
    } else {
        printf("Saved TIFF %s\n", output_path.c_str());
    }
}
