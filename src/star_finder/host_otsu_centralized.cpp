// cpu_otsu_centralized.h
// CPU-only implementation of Otsu threshold with centralization

#ifndef CPU_OTSU_CENTRALIZED_H
#define CPU_OTSU_CENTRALIZED_H

#include <stdint.h>
#include <cmath>

// Constants
#define OTSU_HISTOGRAM_SIZE 256


inline void cpu_calculate_histogram(const u_int16_t *image, u_int64_t npixels, double *histogram) {
    // Initialize histogram
    for (int i = 0; i < OTSU_HISTOGRAM_SIZE; i++) {
        histogram[i] = 0.0;
    }
    
    // Calculate histogram (8-bit values from 16-bit input)
    for (u_int64_t i = 0; i < npixels; i++) {
        // Convert u_int16_t to u_int8_t by dividing by 256
        u_int8_t pixel_value = (u_int8_t)(image[i] / 256);
        histogram[pixel_value] += 1.0;
    }
    
    // Normalize histogram to probabilities
    for (int i = 0; i < OTSU_HISTOGRAM_SIZE; i++) {
        histogram[i] /= (double)npixels;
    }
}

inline u_int8_t cpu_find_otsu_threshold(const double *histogram, u_int64_t npixels) {
    double sum = 0.0;  // Sum of all pixel values * probability
    
    for (int t = 0; t < OTSU_HISTOGRAM_SIZE; t++) {
        sum += t * histogram[t];
    }
    
    double sum_B = 0.0;  // Sum for background class
    double w_B = 0.0;    // Background class probability
    double max_variance = 0.0;
    u_int8_t threshold = 0;
    
    // Find threshold that maximizes between-class variance
    for (int t = 0; t < OTSU_HISTOGRAM_SIZE; t++) {
        w_B += histogram[t];
        
        if (w_B == 0.0)
            continue;
        
        double w_F = 1.0 - w_B;  // Foreground class probability
        if (w_F == 0.0)
            break;
        
        sum_B += t * histogram[t];
        
        double mean_B = sum_B / w_B;
        double mean_F = (sum - sum_B) / w_F;
        
        // Between-class variance
        double variance = w_B * w_F * (mean_B - mean_F) * (mean_B - mean_F);
        
        if (variance > max_variance) {
            max_variance = variance;
            threshold = (u_int8_t)t;
        }
    }
    
    return threshold;
}


inline double cpu_calculate_mean(const u_int16_t *image, u_int64_t width, u_int64_t height) {
    u_int64_t npixels = width * height;
    double sum = 0.0;
    
    for (u_int64_t i = 0; i < npixels; i++) {
        sum += (double)image[i];
    }
    
    return sum / (double)npixels;
}

inline void mean_filter(const u_int16_t *image, double *temp_filtered, 
                        u_int64_t width, u_int64_t height, u_int64_t npixels, 
                        int window_size) {
    // Optimized using integral image (summed area table) - O(width * height)
    // instead of O(width * height * window_size^2)
    
    int half_window = window_size / 2;
    
    // Step 1: Build integral image (summed area table)
    // integral[y][x] = sum of all pixels in rectangle (0,0) to (x,y)
    double *integral = new double[npixels];
    
    // First row
    double row_sum = 0.0;
    for (u_int64_t x = 0; x < width; x++) {
        row_sum += (double)image[x];
        integral[x] = row_sum;
    }
    
    // Remaining rows
    for (u_int64_t y = 1; y < height; y++) {
        row_sum = 0.0;
        u_int64_t row_offset = y * width;
        u_int64_t prev_row_offset = (y - 1) * width;
        for (u_int64_t x = 0; x < width; x++) {
            row_sum += (double)image[row_offset + x];
            integral[row_offset + x] = integral[prev_row_offset + x] + row_sum;
        }
    }
    
    // Step 2: Compute mean for each pixel using integral image in O(1)
    for (u_int64_t y = 0; y < height; y++) {
        for (u_int64_t x = 0; x < width; x++) {
            u_int64_t idx = y * width + x;
            
            // Calculate window bounds
            int64_t y1 = y - half_window;
            if (y1 < 0)
                y1 = 0;

            int64_t y2 = y + half_window;
            if (y2 >= height)
                y2 = height - 1;

            int64_t x1 = x - half_window;
            if (x1 < 0)
                 x1 = 0;
            
            int64_t x2 = x + half_window;
            if (x2 >= width)
                x2 = width - 1;
            
            // Get sum from integral image using inclusion-exclusion
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


void cpu_otsu_centralized_threshold(const u_int16_t *image, u_int8_t *output,
                                    u_int64_t width, u_int64_t height,
                                    u_int16_t window_size, float tr_scale) {
    u_int64_t npixels = width * height;
    
    // 1 - Calculate histogram and find Otsu threshold
    double *histogram = new double[OTSU_HISTOGRAM_SIZE];
    cpu_calculate_histogram(image, npixels, histogram);
    int otsu_threshold = cpu_find_otsu_threshold(histogram, npixels);
    otsu_threshold = otsu_threshold * tr_scale;
    delete[] histogram;
    
    // 2 - Calculate global mean
    double global_mean = cpu_calculate_mean(image, width, height);
    
    // 3 - Create mean filtered image
    double *mean_filtered = new double[npixels];
    mean_filter(image, mean_filtered, width, height, npixels, window_size);
    
    // 4 - Apply centralized threshold
    // T_c[i,j] = M_f[i,j] - t_mean + t_otsu
    for (u_int64_t i = 0; i < npixels; i++) {
        // Convert 16-bit pixel to 8-bit
        int pixel_8bit = image[i] / 256;
        int mean_8bit = global_mean / 256;
        int filtered_8bit = mean_filtered[i] / 256;

        int pixel_threshold = filtered_8bit - mean_8bit + otsu_threshold;
        
        output[i] = (pixel_8bit > pixel_threshold) ? 255 : 0;
    }
    
    delete[] mean_filtered;
}

#endif // CPU_OTSU_CENTRALIZED_H
