#include <stdint.h>
#include <cmath>
#include <float.h>
#include "cuda_helper.h"
#include "common.h"

// Constants
#define OTSU_HISTOGRAM_SIZE  4096
#define OTSU_THREADS_PER_BLOCK 256
#define OTSU_NUM_BLOCKS 32


__global__ void cuda_kernel_find_minmax(const u_int16_t *image, u_int64_t npixels, u_int16_t *block_min, u_int16_t *block_max) {
    extern __shared__ u_int16_t shared_buf[];
    u_int16_t *s_min = shared_buf;
    u_int16_t *s_max = shared_buf + blockDim.x;

    u_int64_t idx = (u_int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    u_int16_t val = (idx < npixels) ? image[idx] : 65535;
    s_min[tid] = val;
    s_max[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_min[tid + s] < s_min[tid]) 
                s_min[tid] = s_min[tid + s];
            if (s_max[tid + s] > s_max[tid])
                s_max[tid] = s_max[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_min[blockIdx.x] = s_min[0];
        block_max[blockIdx.x] = s_max[0];
    }
}

// ---------------------------------------------------------------------------
// 2.  Histogram: map [img_min, img_max] → [0, OTSU_HISTOGRAM_SIZE-1]
// ---------------------------------------------------------------------------
__global__ void cuda_kernel_calculate_histogram(const u_int16_t *image,
                                                u_int64_t npixels,
                                                u_int32_t *histogram,
                                                u_int16_t img_min,
                                                u_int16_t img_max) {
    u_int64_t idx = (u_int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= npixels)
        return;

    u_int16_t v = image[idx];
    if (v < img_min)
        v = img_min;
    if (v > img_max)
        v = img_max;

    // scale to [0, OTSU_HISTOGRAM_SIZE-1]
    int bin = (int)((u_int64_t)(v - img_min) * (OTSU_HISTOGRAM_SIZE - 1)
                    / (img_max - img_min));
    atomicAdd(&histogram[bin], 1);
}

// ---------------------------------------------------------------------------
// 3.  Otsu between-class variance (one thread per threshold candidate)
// ---------------------------------------------------------------------------
__global__ void cuda_kernel_compute_class_variances(const double *histogram,
                                                    double sum_all,
                                                    double *variances) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= OTSU_HISTOGRAM_SIZE)
        return;

    double sum_B = 0.0;
    double w_B   = 0.0;

    for (int i = 0; i <= t; i++) {
        sum_B += (double)i * histogram[i];
        w_B   += histogram[i];
    }

    if (w_B == 0.0 || w_B == 1.0) {
        variances[t] = 0.0;
        return;
    }

    double w_F   = 1.0 - w_B;
    double mean_B = sum_B / w_B;
    double mean_F = (sum_all - sum_B) / w_F;

    variances[t] = w_B * w_F * (mean_B - mean_F) * (mean_B - mean_F);
}

// ---------------------------------------------------------------------------
// 4.  Global mean (parallel reduction)
__global__ void cuda_kernel_calculate_mean(const u_int16_t *image,
                                           u_int64_t npixels,
                                           double *partial_sums) {
    extern __shared__ double shared_sum[];

    u_int64_t idx = (u_int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    shared_sum[tid] = (idx < npixels) ? (double)image[idx] : 0.0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            shared_sum[tid] += shared_sum[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        partial_sums[blockIdx.x] = shared_sum[0];
}

// ---------------------------------------------------------------------------
// 5.  Optimized integral-image kernels
__global__ void cuda_kernel_integral_image_row_pass(const u_int16_t *image,
                                                    double *integral,
                                                    u_int64_t width,
                                                    u_int64_t height) {
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
                                                     u_int64_t width,
                                                     u_int64_t height) {
    u_int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width)
        return;

    double col_sum = 0.0;
    for (u_int64_t y = 0; y < height; y++) {
        col_sum += integral[y * width + x];
        integral[y * width + x] = col_sum;
    }
}

__global__ void cuda_kernel_mean_filter_integral(const double *integral,
                                                  double *temp_filtered,
                                                  u_int64_t width,
                                                  u_int64_t height,
                                                  int half_window) {
    u_int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    u_int64_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int64_t y1 = (int64_t)y - half_window;
    int64_t y2 = (int64_t)y + half_window;
    int64_t x1 = (int64_t)x - half_window;
    int64_t x2 = (int64_t)x + half_window;
    if (y1 < 0)
        y1 = 0;
    if (y2 >= height)
        y2 = height - 1;
    if (x1 < 0)
        x1 = 0;
    if (x2 >= width)
        x2 = width - 1;

    double sum = integral[y2 * width + x2];
    if (y1 > 0)
        sum -= integral[(y1 - 1) * width + x2];
    if (x1 > 0)
        sum -= integral[y2 * width + (x1 - 1)];
    if (y1 > 0 && x1 > 0)
        sum += integral[(y1 - 1) * width + (x1 - 1)];

    double count = (double)((y2 - y1 + 1) * (x2 - x1 + 1));
    temp_filtered[y * width + x] = sum / count;
}

// ---------------------------------------------------------------------------
// 6.  Centralized threshold kernel
__global__ void cuda_kernel_otsu_centralized_threshold(const u_int16_t *image,
                                                       const double *mean_filtered,
                                                       u_int8_t *output,
                                                       u_int64_t npixels,
                                                       double global_mean,
                                                       double otsu_threshold) {
    u_int64_t idx = (u_int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= npixels)
        return;

    double pixel_val    = (double)image[idx];
    double filtered_val = mean_filtered[idx];

    // T_c  =  mean_filtered  -  global_mean  +  otsu_threshold
    double pixel_threshold = filtered_val - global_mean + otsu_threshold;

    output[idx] = (pixel_val > pixel_threshold) ? 255 : 0;
}

// =========================================================================
//  Host-side helper functions

inline void cuda_find_minmax(const u_int16_t *d_image, u_int64_t npixels,
                             u_int16_t &out_min, u_int16_t &out_max) {
    int blocks = (npixels + OTSU_THREADS_PER_BLOCK - 1) / OTSU_THREADS_PER_BLOCK;

    u_int16_t *d_block_min, *d_block_max;
    CHECK(cudaMalloc(&d_block_min, sizeof(u_int16_t) * blocks));
    CHECK(cudaMalloc(&d_block_max, sizeof(u_int16_t) * blocks));

    cuda_kernel_find_minmax<<<blocks, OTSU_THREADS_PER_BLOCK, 2 * OTSU_THREADS_PER_BLOCK * sizeof(u_int16_t)>>>(
        d_image, npixels, d_block_min, d_block_max);
    CHECK(cudaDeviceSynchronize());

    u_int16_t *h_min = new u_int16_t[blocks];
    u_int16_t *h_max = new u_int16_t[blocks];
    CHECK(cudaMemcpy(h_min, d_block_min, sizeof(u_int16_t) * blocks, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_max, d_block_max, sizeof(u_int16_t) * blocks, cudaMemcpyDeviceToHost));

    out_min = 65535; out_max = 0;
    for (int i = 0; i < blocks; i++) {
        if (h_min[i] < out_min)
            out_min = h_min[i];
        if (h_max[i] > out_max)
            out_max = h_max[i];
    }
    delete[] h_min;
    delete[] h_max;
    CHECK(cudaFree(d_block_min));
    CHECK(cudaFree(d_block_max));
}

inline double* cuda_calculate_histogram(const u_int16_t *d_image,
                                        u_int64_t npixels,
                                        u_int16_t img_min,
                                        u_int16_t img_max) {
    // calculate the histogram (done on GPU)
    u_int32_t *d_histogram;
    CHECK(cudaMalloc(&d_histogram, sizeof(u_int32_t) * OTSU_HISTOGRAM_SIZE));
    CHECK(cudaMemset(d_histogram, 0, sizeof(u_int32_t) * OTSU_HISTOGRAM_SIZE));

    int blocks = (npixels + OTSU_THREADS_PER_BLOCK - 1) / OTSU_THREADS_PER_BLOCK;
    cuda_kernel_calculate_histogram<<<blocks, OTSU_THREADS_PER_BLOCK>>>(
        d_image, npixels, d_histogram, img_min, img_max);

    // Allocate memory and copy histogram to host
    u_int32_t *h_histogram = new u_int32_t[OTSU_HISTOGRAM_SIZE];
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_histogram, d_histogram, sizeof(u_int32_t) * OTSU_HISTOGRAM_SIZE, cudaMemcpyDeviceToHost));
    // Free device histogram memory
    CHECK(cudaFree(d_histogram));

    // normalize histogram to [0,1] ( -> probability distribution)
    double *normalized = new double[OTSU_HISTOGRAM_SIZE];
    for (int i = 0; i < OTSU_HISTOGRAM_SIZE; i++)
        normalized[i] = (double)h_histogram[i] / (double)npixels;

    delete[] h_histogram;
    return normalized;
}

// return bin index of otsu threshold (0..OTSU_HISTOGRAM_SIZE-1)
inline int cuda_find_otsu_threshold(const double *histogram) {
    double *d_histogram;
    CHECK(cudaMalloc(&d_histogram, sizeof(double) * OTSU_HISTOGRAM_SIZE));
    CHECK(cudaMemcpy(d_histogram, histogram, sizeof(double) * OTSU_HISTOGRAM_SIZE, cudaMemcpyHostToDevice));

    double *d_variances;
    CHECK(cudaMalloc(&d_variances, sizeof(double) * OTSU_HISTOGRAM_SIZE));

    // expected bin value
    double sum_all = 0.0;
    for (int i = 0; i < OTSU_HISTOGRAM_SIZE; i++)
        sum_all += (double)i * histogram[i];

    int blocks = (OTSU_HISTOGRAM_SIZE + OTSU_THREADS_PER_BLOCK - 1) / OTSU_THREADS_PER_BLOCK;
    cuda_kernel_compute_class_variances<<<blocks, OTSU_THREADS_PER_BLOCK>>>(
        d_histogram, sum_all, d_variances);
    CHECK(cudaDeviceSynchronize());

    double *h_variances = new double[OTSU_HISTOGRAM_SIZE];
    CHECK(cudaMemcpy(h_variances, d_variances, sizeof(double) * OTSU_HISTOGRAM_SIZE, cudaMemcpyDeviceToHost));

    double max_variance = 0.0;
    int threshold_bin = 0;
    for (int t = 0; t < OTSU_HISTOGRAM_SIZE; t++) {
        if (h_variances[t] > max_variance) {
            max_variance = h_variances[t];
            threshold_bin = t;
        }
    }

    delete[] h_variances;
    CHECK(cudaFree(d_histogram));
    CHECK(cudaFree(d_variances));

    return threshold_bin;
}

// bin index -> threshold value
inline double bin_to_value(int bin, u_int16_t img_min, u_int16_t img_max) {
    return (double)img_min
           + (double)bin * (double)(img_max - img_min)
               / (double)(OTSU_HISTOGRAM_SIZE - 1);
}

// global mean (parallel reduction)
inline double cuda_calculate_mean(const u_int16_t *d_image, u_int64_t npixels) {
    int blocks = (npixels + OTSU_THREADS_PER_BLOCK - 1) / OTSU_THREADS_PER_BLOCK;

    double *d_partial_sums;
    CHECK(cudaMalloc(&d_partial_sums, sizeof(double) * blocks));

    cuda_kernel_calculate_mean<<<blocks, OTSU_THREADS_PER_BLOCK, sizeof(double) * OTSU_THREADS_PER_BLOCK>>>(
        d_image, npixels, d_partial_sums);
    CHECK(cudaDeviceSynchronize());

    double *h_partial = new double[blocks];
    CHECK(cudaMemcpy(h_partial, d_partial_sums, sizeof(double) * blocks, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_partial_sums));

    double total = 0.0;
    for (int i = 0; i < blocks; i++)
        total += h_partial[i];
    delete[] h_partial;

    return total / (double)npixels;
}

// =========================================================================
//  Host-side main function
void cuda_otsu_centralized_threshold(const u_int16_t *d_image,
                                     u_int8_t *d_output,
                                     u_int64_t width,
                                     u_int64_t height,
                                     int window_size,
                                     float th_scale) {
    u_int64_t npixels = width * height;

    // 1. find image min and max
    u_int16_t img_min, img_max;
    cuda_find_minmax(d_image, npixels, img_min, img_max);

    // 1.1 case of uniform image, avoid / 0 
    if (img_max <= img_min) 
        img_max = img_min + 1;

    // 2. histogram and find threshold (bin index)
    double *histogram = cuda_calculate_histogram(d_image, npixels,
                                                 img_min, img_max);
    int otsu_bin = cuda_find_otsu_threshold(histogram);
    delete[] histogram;

    // 2.1 get threshold value from bin index
    double otsu_threshold = bin_to_value(otsu_bin, img_min, img_max);
    otsu_threshold *= (double)th_scale;

    // 3. global mean
    double global_mean = cuda_calculate_mean(d_image, npixels);

    // 4. mean filter with integral optimizations
    double *d_integral;
    CHECK(cudaMalloc(&d_integral, sizeof(double) * npixels));

    int row_blocks = (height + 255) / 256;
    cuda_kernel_integral_image_row_pass<<<row_blocks, 256>>>(
        d_image, d_integral, width, height);
    CHECK(cudaDeviceSynchronize());

    int col_blocks = (width + 255) / 256;
    cuda_kernel_integral_image_col_pass<<<col_blocks, 256>>>(
        d_integral, width, height);
    CHECK(cudaDeviceSynchronize());

    double *d_mean_filtered;
    CHECK(cudaMalloc(&d_mean_filtered, sizeof(double) * npixels));

    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    cuda_kernel_mean_filter_integral<<<grid_size, block_size>>>(
        d_integral, d_mean_filtered, width, height, (int)(window_size / 2));
    CHECK(cudaDeviceSynchronize());

    // 5. thresholding
    int blocks = (npixels + OTSU_THREADS_PER_BLOCK - 1) / OTSU_THREADS_PER_BLOCK;
    cuda_kernel_otsu_centralized_threshold<<<blocks, OTSU_THREADS_PER_BLOCK>>>(
        d_image, d_mean_filtered, d_output, npixels, global_mean, otsu_threshold);
    CHECK(cudaDeviceSynchronize());

    // 6. memory cleanup
    CHECK(cudaFree(d_integral));
    CHECK(cudaFree(d_mean_filtered));
}
