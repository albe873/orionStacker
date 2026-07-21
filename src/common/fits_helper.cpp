#include <stdio.h>
#include <fitsio.h>
#include <string.h>
#include <time.h>
#include <string>
#include <dirent.h>
#include <algorithm>
#include <vector>
#include <utility>
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


double get_fits_date_avg(fitsfile *fptr) {
    int status = 0;
    char date_str[128];
    // Prima prova DATE-AVG, poi DATE-OBS come fallback
    if (fits_read_key(fptr, TSTRING, "DATE-AVG", date_str, NULL, &status)) {
        status = 0; // reset status
        if (fits_read_key(fptr, TSTRING, "DATE-OBS", date_str, NULL, &status)) {
            // Nessun campo data trovato
            return 0.0;
        }
    }

    // Il formato tipico è: "2024-08-11T00:20:54.000" oppure "2024-08-11T00:20:54"
    struct tm tm_val;
    memset(&tm_val, 0, sizeof(tm_val));

    double frac_sec = 0.0;
    // Prova a parsare con frazione di secondo
    int matched = sscanf(date_str, "%d-%d-%dT%d:%d:%lf",
                         &tm_val.tm_year, &tm_val.tm_mon, &tm_val.tm_mday,
                         &tm_val.tm_hour, &tm_val.tm_min, &frac_sec);
    if (matched < 5) {
        // Prova formato con data e ora separati da spazio
        matched = sscanf(date_str, "%d-%d-%d %d:%d:%lf",
                         &tm_val.tm_year, &tm_val.tm_mon, &tm_val.tm_mday,
                         &tm_val.tm_hour, &tm_val.tm_min, &frac_sec);
    }
    if (matched < 5) {
        fprintf(stderr, "get_fits_date_avg: cannot parse date string '%s'\n", date_str);
        return 0.0;
    }

    tm_val.tm_year -= 1900;  // tm_year è years since 1900
    tm_val.tm_mon  -= 1;     // tm_mon è 0-based
    tm_val.tm_sec   = (int)frac_sec;
    tm_val.tm_isdst = -1;    // lascia che il sistema decida DST

    time_t t = timegm(&tm_val);  // UTC, non locale
    if (t == (time_t)-1) {
        fprintf(stderr, "get_fits_date_avg: timegm failed for '%s'\n", date_str);
        return 0.0;
    }

    return (double)t + (frac_sec - (int)frac_sec);
}


int find_mid_image_index(const double *timestamps, int count) {
    if (count <= 0) return -1;
    if (count == 1) return 0;

    // Crea coppie (timestamp, indice_originale) e ordina per timestamp
    std::vector<std::pair<double, int>> indexed;
    indexed.reserve(count);
    for (int i = 0; i < count; ++i) {
        indexed.emplace_back(timestamps[i], i);
    }

    // Ordina per timestamp crescente
    std::sort(indexed.begin(), indexed.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    // Restituisce l'indice originale dell'elemento mediano
    return indexed[count / 2].second;
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




int check_directory(const char *dir_path, int *count, long *width, long *height, long *n_chan, long expect_n_chan = 0) {
    DIR *dir = opendir(dir_path);
    if (!dir) {
        perror("opendir");
        return 1;
    }

    struct dirent *entry;
    int status=0;

    // Conta e misura le immagini
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type != DT_REG) continue;

        if (strstr(entry->d_name, ".fits") || strstr(entry->d_name, ".fit")) {
            char path[1024];
            snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);

            fitsfile *fptr = nullptr;
            open_fits(path, &fptr);
            long w,h,n;
            get_fits_dimensions(fptr, &w,&h,&n);
            if (expect_n_chan > 0 && n != expect_n_chan) {
                fprintf(stderr,"Skipping %s: expected %ld channel\n", path, expect_n_chan);
                fits_close_file(fptr,&status);
                continue;
            }
            if (*count == 0) {
                *width=w;
                *height=h;
                *n_chan=n;
            }
            else if (w != *width || h != *height) {
                fprintf(stderr,"Skipping %s: dimensions mismatch\n", path);
                fits_close_file(fptr,&status);
                continue;
            }
            fits_close_file(fptr,&status);
            (*count)++;
        }
    }
    closedir(dir);

    if (*count == 0) {
        fprintf(stderr,"No valid images\n");
        return 1;
    }
    printf("  Found %d images\n", *count);
    return 0;
}

// Rileggi le immagini e copia in memoria chiamando funzione esterna
int load_images_to_memory(const char *dir_path, u_int16_t *img_all, long width, long height, long n_chan, int count, double *timestamps) {
    DIR *dir = opendir(dir_path);
    if (!dir) {
        perror("opendir");
        return 1;
    }

    struct dirent *entry;
    int status=0, idx=0;
    long w, h, n;
    long data_size = width * height * n_chan;
    while ((entry = readdir(dir)) != NULL && idx<count) {
        if (entry->d_type != DT_REG)
            continue;
        if (!(strstr(entry->d_name, ".fits") || strstr(entry->d_name, ".fit")))
            continue;

        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);

        fitsfile *fptr = nullptr;
        open_fits(path, &fptr);

        get_fits_dimensions(fptr, &w,&h,&n);
        if (w != width || h != height || n != n_chan) {
            fits_close_file(fptr, &status);
            continue;
        }

        if (timestamps != nullptr) {
            timestamps[idx] = get_fits_date_avg(fptr);
        }

        get_fits_data(fptr, data_size, img_all + idx*data_size);
        fits_close_file(fptr,&status);

        idx++;
    }
    if (idx != count)
        printf("  Warning! Number of expected images: %d, Actually loaded: %d", count, idx);
    closedir(dir);
    return 0;
}


bool find_latest_master_file(const string &dir_path, const string &master_type, string &file_path) {
    string prefix = "master_" + master_type + "_";
    DIR *dir = opendir(dir_path.c_str());
    if (!dir) {
        return false;
    }

    struct dirent *entry;
    string latest_file;
    string latest_ts;

    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type != DT_REG)
            continue;
        string name(entry->d_name);
        // Must start with the prefix and end with .fits
        if (name.rfind(prefix, 0) != 0)
            continue;
        if (name.size() < 4)
            continue;
        // Check .fits extension
        string lower_name = name;
        transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
        if (lower_name.substr(lower_name.size() - 5) != ".fits")
            continue;
        // Extract the timestamp: after prefix we expect "YYYYMMDD_HHMMSS.fits"
        string rest = name.substr(prefix.size());
        if (rest.size() < 20) continue;  // "YYYYMMDD_HHMMSS.fits" = 20 chars
        string ts = rest.substr(0, 15);   // "YYYYMMDD_HHMMSS" = 15 chars
        if (ts > latest_ts) {
            latest_ts = ts;
            latest_file = entry->d_name;
        }
    }
    closedir(dir);

    if (latest_file.empty()) {
        return false;
    }

    file_path = dir_path;
    if (file_path.back() != '/')
        file_path += "/";
    file_path += latest_file;

    printf("Found existing master %s: %s\n", master_type.c_str(), file_path.c_str());
    return true;
}
