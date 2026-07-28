#include <stdint.h>
#include <cmath>
#include <algorithm>

#define OTSU_HISTOGRAM_SIZE 65536


inline void cpu_calculate_histogram(const uint16_t *image, uint64_t npixels, double *histogram) {
    for (int i = 0; i < OTSU_HISTOGRAM_SIZE; i++)
        histogram[i] = 0.0;

    for (uint64_t i = 0; i < npixels; i++) {
        auto v = image[i];
        histogram[v] += 1.0;
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

    // 1 - histogram + Otsu
    double *histogram = new double[OTSU_HISTOGRAM_SIZE];
    cpu_calculate_histogram(image, npixels, histogram);
    double otsu_threshold = cpu_find_otsu_threshold(histogram);
    otsu_threshold *= (double)tr_scale;
    delete[] histogram;

    // 2 - global mean
    double global_mean = cpu_calculate_mean(image, width, height);

    // 3 - local mean via integral image
    double *mean_filtered = new double[npixels];
    mean_filter(image, mean_filtered, width, height, npixels, window_size);

    // 4 - centralized threshold
    for (uint64_t i = 0; i < npixels; i++) {
        float pixel_val       = image[i];
        float filtered_val    = mean_filtered[i];
        float pixel_threshold = filtered_val - global_mean + otsu_threshold;
        output[i] = (pixel_val > pixel_threshold) ? 255 : 0;
    }

    delete[] mean_filtered;
}
