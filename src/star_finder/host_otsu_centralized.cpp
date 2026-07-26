#include <stdint.h>
#include <cmath>
#include <algorithm>

#define OTSU_HISTOGRAM_SIZE 4096


// ---------- find global min / max ----------
inline void cpu_find_minmax(const uint16_t *image, uint64_t npixels,
                            uint16_t &out_min, uint16_t &out_max) {
    out_min = 65535;
    out_max = 0;
    #pragma omp parallel for shared(out_min, out_max)
    for (uint64_t i = 0; i < npixels; i++) {
        if (image[i] < out_min)
            out_min = image[i];
        if (image[i] > out_max)
            out_max = image[i];
    }
}


inline void cpu_calculate_histogram(const uint16_t *image, uint64_t npixels,
                                    double *histogram,
                                    uint16_t img_min, uint16_t img_max) {
    for (int i = 0; i < OTSU_HISTOGRAM_SIZE; i++)
        histogram[i] = 0.0;

    double scale = (double)(OTSU_HISTOGRAM_SIZE - 1) / (double)(img_max - img_min);

    for (uint64_t i = 0; i < npixels; i++) {
        auto v = image[i];
        if (v < img_min)
            v = img_min;
        if (v > img_max)
            v = img_max;
        int bin = (int)((double)(v - img_min) * scale);
        histogram[bin] += 1.0;
    }

    // normalize to probabilities
    for (int i = 0; i < OTSU_HISTOGRAM_SIZE; i++)
        histogram[i] /= (double)npixels;
}


inline int cpu_find_otsu_threshold(const double *histogram) {
    double sum_all = 0.0;
    for (int i = 0; i < OTSU_HISTOGRAM_SIZE; i++)
        sum_all += (double)i * histogram[i];

    double sum_B = 0.0, w_B = 0.0;
    double max_variance = 0.0;
    int threshold_bin = 0;

    for (int t = 0; t < OTSU_HISTOGRAM_SIZE; t++) {
        w_B += histogram[t];
        if (w_B == 0.0)
            continue;

        double w_F = 1.0 - w_B;
        if (w_F != 0.0) {

            sum_B += (double)t * histogram[t];

            double mean_B = sum_B / w_B;
            double mean_F = (sum_all - sum_B) / w_F;
            double variance = w_B * w_F * (mean_B - mean_F) * (mean_B - mean_F);

            if (variance > max_variance) {
                max_variance = variance;
                threshold_bin = t;
            }
        }
    }
    return threshold_bin;
}


// ---------- map OTSU bin index back to 16-bit value ----------
inline double bin_to_value(int bin, uint16_t img_min, uint16_t img_max) {
    return (double)img_min
           + (double)bin * (double)(img_max - img_min)
               / (double)(OTSU_HISTOGRAM_SIZE - 1);
}


inline double cpu_calculate_mean(const uint16_t *image, uint64_t width, uint64_t height) {
    uint64_t npixels = width * height;
    double sum = 0.0;
    for (uint64_t i = 0; i < npixels; i++)
        sum += (double)image[i];
    return sum / (double)npixels;
}


inline void mean_filter(const uint16_t *image, double *temp_filtered,
                        uint64_t width, uint64_t height, uint64_t npixels,
                        int window_size) {
    int half_window = window_size / 2;

    // 1. integral image
    double *integral = new double[npixels];

    double row_sum = 0.0;
    #pragma omp parallel for reduction(+:row_sum)
    for (uint64_t x = 0; x < width; x++) {
        row_sum += (double)image[x];
        integral[x] = row_sum;
    }
    for (uint64_t y = 1; y < height; y++) {
        row_sum = 0.0;
        uint64_t row_offset = y * width;
        uint64_t prev_row_offset = (y - 1) * width;
        for (uint64_t x = 0; x < width; x++) {
            row_sum += (double)image[row_offset + x];
            integral[row_offset + x] = integral[prev_row_offset + x] + row_sum;
        }
    }

    // 2. O(1) mean per pixel
    #pragma omp parallel for
    for (uint64_t y = 0; y < height; y++) {
        for (uint64_t x = 0; x < width; x++) {
            uint64_t idx = y * width + x;

            int64_t y1 = y - half_window;
            if (y1 < 0)
                y1 = 0;
            int64_t y2 = y + half_window;
            if (y2 >= (int64_t)height)
                y2 = height - 1;
            int64_t x1 = x - half_window;
            if (x1 < 0)
                x1 = 0;
            int64_t x2 = x + half_window;
            if (x2 >= (int64_t)width) 
                x2 = width - 1;

            double sum = integral[y2 * width + x2];
            if (y1 > 0)
                sum -= integral[(y1 - 1) * width + x2];
            if (x1 > 0)
                sum -= integral[y2 * width + (x1 - 1)];
            if (y1 > 0 && x1 > 0)
                sum += integral[(y1 - 1) * width + (x1 - 1)];

            double count = (double)((y2 - y1 + 1) * (x2 - x1 + 1));
            temp_filtered[idx] = sum / count;
        }
    }
    delete[] integral;
}


void cpu_otsu_centralized_threshold(const uint16_t *image, uint8_t *output,
                                    uint64_t width, uint64_t height,
                                    uint16_t window_size, float tr_scale) {
    uint64_t npixels = width * height;

    // 0 - find image min / max
    uint16_t img_min, img_max;
    cpu_find_minmax(image, npixels, img_min, img_max);
    if (img_max <= img_min) img_max = img_min + 1;

    // 1 - histogram + Otsu (bin index)
    double *histogram = new double[OTSU_HISTOGRAM_SIZE];
    cpu_calculate_histogram(image, npixels, histogram, img_min, img_max);
    int otsu_bin = cpu_find_otsu_threshold(histogram);
    delete[] histogram;

    // bin index -> threshold value
    double otsu_threshold = bin_to_value(otsu_bin, img_min, img_max);
    otsu_threshold *= (double)tr_scale;

    // 2 - global mean
    double global_mean = cpu_calculate_mean(image, width, height);

    // 3 - local mean via integral image
    double *mean_filtered = new double[npixels];
    mean_filter(image, mean_filtered, width, height, npixels, window_size);

    // 4 - centralized threshold
    for (uint64_t i = 0; i < npixels; i++) {
        double pixel_val       = (double)image[i];
        double filtered_val    = mean_filtered[i];
        double pixel_threshold = filtered_val - global_mean + otsu_threshold;
        output[i] = (pixel_val > pixel_threshold) ? 255 : 0;
    }

    delete[] mean_filtered;
}
