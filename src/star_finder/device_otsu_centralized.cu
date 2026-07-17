// device_otsu_centralized.h
// GPU (CUDA/HIP) implementation of Otsu threshold with centralization
// Header-only library - works with both native CUDA and HIP/HIPIFY

#ifndef CUDA_OTSU_CENTRALIZED_CU
#define CUDA_OTSU_CENTRALIZED_CU

#include <stdint.h>
#include <cmath>
#include "cuda_helper.h"
#include "common.h"

// Constants
#define OTSU_HISTOGRAM_SIZE 256
#define OTSU_THREADS_PER_BLOCK 256
#define OTSU_NUM_BLOCKS 32


__global__ void cuda_kernel_calculate_histogram(const u_int16_t *image, u_int64_t npixels, u_int32_t *histogram) {
    u_int64_t idx = (u_int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < npixels) {
        // scaling u_int16_t to u_int8_t
        u_int8_t pixel_value = (u_int8_t)(image[idx] / 256);
        // needs to perform an atomic add to avoid race conditions when multiple threads
        atomicAdd(&histogram[pixel_value], 1);
    }
}

__global__ void cuda_kernel_compute_class_variances(const double *histogram, 
                                                    double sum_all,
                                                    u_int64_t npixels,
                                                    double *variances) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (t >= OTSU_HISTOGRAM_SIZE) // -> 256 max parallel threads
        return;
    
    double sum_B = 0.0;
    double w_B = 0.0;
    
    // Calculate for background class (0 to t)
    for (int i = 0; i <= t; i++) {
        sum_B += i * histogram[i];
        w_B += histogram[i];
    }
    
    if (w_B == 0.0 || w_B == 1.0) {
        variances[t] = 0.0;
        return;
    }
    
    double w_F = 1.0 - w_B;
    double mean_B = sum_B / w_B;
    double mean_F = (sum_all - sum_B) / w_F;
    
    // Between-class variance
    variances[t] = w_B * w_F * (mean_B - mean_F) * (mean_B - mean_F);
}


__global__ void cuda_kernel_calculate_mean(const u_int16_t *image, u_int64_t npixels,
                                           double *partial_sums) {
    extern __shared__ double shared_sum[];
    
    u_int64_t idx = (u_int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Load and sum
    shared_sum[tid] = 0.0;
    if (idx < npixels) {
        shared_sum[tid] = (double)image[idx];
    }
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    // Write block result
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_sum[0];
    }
}


__global__ void cuda_kernel_otsu_centralized_threshold(const u_int16_t *image,
                                                       u_int8_t *output,
                                                       u_int64_t npixels,
                                                       u_int8_t otsu_threshold) {
    u_int64_t idx = (u_int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < npixels) {
        // Convert 16-bit pixel to 8-bit
        u_int8_t pixel_8bit = (u_int8_t)(image[idx] / 256);
        
        output[idx] = (pixel_8bit > otsu_threshold) ? 255 : 0;
    }
}


__global__ void cuda_kernel_local_mean_filter(const u_int16_t *image,
                                              double *temp_filtered,
                                              u_int64_t width, u_int64_t height,
                                              int half_window) {
    u_int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    u_int64_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height)
        return;
    
    u_int64_t idx = y * width + x;
    
    double sum = 0.0;
    double count = 0.0;
    
    // Apply local mean filter
    for (int wy = -half_window; wy <= half_window; wy++) {
        int64_t ny = (int64_t)y + wy;
        if (ny < 0 || ny >= height)
            continue;
        
        for (int wx = -half_window; wx <= half_window; wx++) {
            int64_t nx = (int64_t)x + wx;
            if (nx < 0 || nx >= (int64_t)width)
                continue;
            
            sum += (double)image[ny * width + nx];
            count += 1.0;
        }
    }
    
    temp_filtered[idx] = (count > 0) ? (sum / count) : (double)image[idx];
}


// Optimized integral image (summed area table) kernels
__global__ void cuda_kernel_integral_image_row_pass(const u_int16_t *image,
                                                    double *integral,
                                                    u_int64_t width, u_int64_t height) {
    u_int64_t y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= height)
        return;
    
    u_int64_t row_offset = y * width;
    double row_sum = 0.0;
    
    for (u_int64_t x = 0; x < width; x++) {
        row_sum += (double)image[row_offset + x];
        integral[row_offset + x] = row_sum;
    }
}

__global__ void cuda_kernel_integral_image_col_pass(double *integral,
                                                     u_int64_t width, u_int64_t height) {
    u_int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) return;
    
    double col_sum = 0.0;
    for (u_int64_t y = 0; y < height; y++) {
        col_sum += integral[y * width + x];
        integral[y * width + x] = col_sum;
    }
}

__global__ void cuda_kernel_mean_filter_integral(const double *integral,
                                                  double *temp_filtered,
                                                  u_int64_t width, u_int64_t height,
                                                  int half_window) {
    u_int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    u_int64_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height)
        return;
    
    u_int64_t idx = y * width + x;
    
    // Calculate window bounds
    int64_t y1 = (int64_t)y - half_window;
    int64_t y2 = (int64_t)y + half_window;
    int64_t x1 = (int64_t)x - half_window;
    int64_t x2 = (int64_t)x + half_window;
    
    // Clamp to image boundaries
    if (y1 < 0)
        y1 = 0;
    if (y2 >= (int64_t)height)
        y2 = height - 1;
    if (x1 < 0)
        x1 = 0;
    if (x2 >= (int64_t)width)
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

/**
 * CUDA kernel: Apply centralized threshold with local mean filtering
 */
__global__ void cuda_kernel_otsu_centralized_threshold(const u_int16_t *image,
                                                       const double *temp_filtered,
                                                       u_int8_t *output,
                                                       u_int64_t npixels,
                                                       double global_mean,
                                                       int otsu_threshold) {
    u_int64_t idx = (u_int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < npixels) {
        // Convert 16-bit pixel to 8-bit
        int pixel_8bit = image[idx] / 256;
        int mean_8bit = global_mean / 256;
        int filtered_8bit = temp_filtered[idx] / 256;

        // Centralized threshold: T_c = filtered_mean - global_mean + otsu_threshold
        int pixel_threshold = filtered_8bit - mean_8bit + otsu_threshold;
        
        output[idx] = (pixel_8bit > pixel_threshold) ? 255 : 0;
    }
}

// -----------------------------------------------------------------------------
// Helper functions to call kernels and manage memory (CPU side)


inline double* cuda_calculate_histogram(const u_int16_t *d_image, u_int64_t npixels) {
    u_int32_t *d_histogram;
    CHECK(cudaMalloc(&d_histogram, sizeof(u_int32_t) * OTSU_HISTOGRAM_SIZE));
    CHECK(cudaMemset(d_histogram, 0, sizeof(u_int32_t) * OTSU_HISTOGRAM_SIZE));
    
    int blocks = (npixels + OTSU_THREADS_PER_BLOCK - 1) / OTSU_THREADS_PER_BLOCK;
    cuda_kernel_calculate_histogram<<<blocks, OTSU_THREADS_PER_BLOCK>>>(
        d_image, npixels, d_histogram);
    
    CHECK(cudaDeviceSynchronize());
    
    // Copy histogram to host
    u_int32_t *h_histogram = new u_int32_t[OTSU_HISTOGRAM_SIZE];
    CHECK(cudaMemcpy(h_histogram, d_histogram, sizeof(u_int32_t) * OTSU_HISTOGRAM_SIZE,
               cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_histogram));
    
    // Normalize to probabilities
    double *normalized = new double[OTSU_HISTOGRAM_SIZE];
    for (int i = 0; i < OTSU_HISTOGRAM_SIZE; i++) {
        normalized[i] = (double)h_histogram[i] / (double)npixels;
    }
    delete[] h_histogram;
    
    return normalized;
}

inline int cuda_find_otsu_threshold(const double *histogram, u_int64_t npixels) {
    double *d_histogram;
    CHECK(cudaMalloc(&d_histogram, sizeof(double) * OTSU_HISTOGRAM_SIZE));
    CHECK(cudaMemcpy(d_histogram, histogram, sizeof(double) * OTSU_HISTOGRAM_SIZE, cudaMemcpyHostToDevice));
    
    double *d_variances;
    CHECK(cudaMalloc(&d_variances, sizeof(double) * OTSU_HISTOGRAM_SIZE));
    
    // Calculate sum for Otsu
    double sum = 0.0;
    for (int i = 0; i < OTSU_HISTOGRAM_SIZE; i++) {
        sum += i * histogram[i];
    }
    
    // Compute variances
    int blocks = (OTSU_HISTOGRAM_SIZE + OTSU_THREADS_PER_BLOCK - 1) / OTSU_THREADS_PER_BLOCK;
    cuda_kernel_compute_class_variances<<<blocks, OTSU_THREADS_PER_BLOCK>>>(
        d_histogram, sum, npixels, d_variances);
    
    CHECK(cudaDeviceSynchronize());
    
    // Find maximum variance on host
    double *h_variances = new double[OTSU_HISTOGRAM_SIZE];
    CHECK(cudaMemcpy(h_variances, d_variances, sizeof(double) * OTSU_HISTOGRAM_SIZE, cudaMemcpyDeviceToHost));
    
    double max_variance = 0.0;
    int threshold = 0;
    for (int t = 0; t < OTSU_HISTOGRAM_SIZE; t++) {
        if (h_variances[t] > max_variance) {
            max_variance = h_variances[t];
            threshold = t;
        }
    }
    
    delete[] h_variances;
    CHECK(cudaFree(d_histogram));
    CHECK(cudaFree(d_variances));
    
    return threshold;
}

inline double cuda_calculate_mean(const u_int16_t *d_image, u_int64_t npixels) {
    int blocks = (npixels + OTSU_THREADS_PER_BLOCK - 1) / OTSU_THREADS_PER_BLOCK;
    
    double *d_partial_sums;
    CHECK(cudaMalloc(&d_partial_sums, sizeof(double) * blocks));
    
    cuda_kernel_calculate_mean<<<blocks, OTSU_THREADS_PER_BLOCK, 
                                  sizeof(double) * OTSU_THREADS_PER_BLOCK>>>(
        d_image, npixels, d_partial_sums);
    
    CHECK(cudaDeviceSynchronize());
    
    // Copy partial sums to host and finish reduction
    double *h_partial_sums = new double[blocks];
    CHECK(cudaMemcpy(h_partial_sums, d_partial_sums, sizeof(double) * blocks,
               cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_partial_sums));
    
    double total_sum = 0.0;
    for (int i = 0; i < blocks; i++) {
        total_sum += h_partial_sums[i];
    }
    delete[] h_partial_sums;
    
    return total_sum / (double)npixels;
}


// -----------------------------------------------------------------------------
// Complete functions to call 

void cuda_otsu_centralized_threshold(const u_int16_t *d_image, u_int8_t *d_output,
                                           u_int64_t width, u_int64_t height,
                                           int window_size, float th_scale) {
    u_int64_t npixels = width * height;
    
    // 1 - Calculate histogram and find Otsu threshold
    double *histogram = cuda_calculate_histogram(d_image, npixels);
    int otsu_threshold = cuda_find_otsu_threshold(histogram, npixels);
    otsu_threshold = otsu_threshold * th_scale;
    delete[] histogram;
    
    // 2 - Calculate global mean
    double global_mean = cuda_calculate_mean(d_image, npixels);
    
    // 3 - Apply optimized local mean filtering using integral image
    double *d_integral;
    CHECK(cudaMalloc(&d_integral, sizeof(double) * npixels));
    
    // 3.1 - row-wise prefix sum
    int row_blocks = (height + 255) / 256;
    cuda_kernel_integral_image_row_pass<<<row_blocks, 256>>>(d_image, d_integral, width, height);
    CHECK(cudaDeviceSynchronize());
    
    // 3.2 - column-wise prefix sum
    int col_blocks = (width + 255) / 256;
    cuda_kernel_integral_image_col_pass<<<col_blocks, 256>>>(d_integral, width, height);
    CHECK(cudaDeviceSynchronize());
    
    // 4 - mean filter using integral image
    double *d_mean_filtered;
    CHECK(cudaMalloc(&d_mean_filtered, sizeof(double) * npixels));
    
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    cuda_kernel_mean_filter_integral<<<grid_size, block_size>>>(
        d_integral, d_mean_filtered, width, height, (int) (window_size/2));
    CHECK(cudaDeviceSynchronize());
    
    // 5 - centralized threshold
    int blocks = (npixels + OTSU_THREADS_PER_BLOCK - 1) / OTSU_THREADS_PER_BLOCK;
    cuda_kernel_otsu_centralized_threshold<<<blocks, OTSU_THREADS_PER_BLOCK>>>(
        d_image, d_mean_filtered, d_output, npixels, global_mean, otsu_threshold);
    CHECK(cudaDeviceSynchronize());

    // 6 - memory cleanup
    CHECK(cudaFree(d_integral));
    CHECK(cudaFree(d_mean_filtered));
}

#endif // CUDA_OTSU_CENTRALIZED_CU
