#include <stdint.h>
#include <cmath>
#include <float.h>
#include "cuda_helper.hh"
#include "common.hh"

// Constants
#define OTSU_HISTOGRAM_SIZE  65536
#define OTSU_THREADS_PER_BLOCK 256
#define OTSU_NUM_BLOCKS 32


// ---------------------------------------------------------------------------
// Histogram: map [img_min, img_max] → [0, OTSU_HISTOGRAM_SIZE-1]
__global__ void kernel_calculate_histogram(const uint16_t *image, uint64_t npixels, uint32_t *histogram) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= npixels)
        return;

    auto v = image[idx];
    atomicAdd(&histogram[v], 1);
}

// ---------------------------------------------------------------------------
// Prefix scan
//     I don't want that each thread loops to all the histogram to compute 
//     prefix_w[t]   = sum from i=0 to t of histogram[i]
//     prefix_sum[t] = sum from i=0 to t of i * histogram[i]
//     so I compute a prefix P[0] = A[0]
//                           P[1] = A[0] + A[1]
//     and I have all the sums, I don't need to recompute them in the variance kernel
__global__ void kernel_prefix_scan(const double *histogram,
                                   double *prefix_w,
                                   double *prefix_sum,
                                   double *block_w_totals,
                                   double *block_sum_totals) {
    extern __shared__ double scan_shared_buf[];
    double *s_w   = scan_shared_buf;
    double *s_sum = scan_shared_buf + blockDim.x;

    int t   = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int tid = (int)threadIdx.x;

    s_w[tid]   = (t < OTSU_HISTOGRAM_SIZE) ? histogram[t] : 0.0;
    s_sum[tid] = (t < OTSU_HISTOGRAM_SIZE) ? (double)t * histogram[t] : 0.0;
    __syncthreads();
    
    // Hillis-Steele parallel prefix sum
    // every thread
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        double w_add = 0.0, sum_add = 0.0;
        if (tid >= offset) {
            w_add   = s_w[tid - offset];
            sum_add = s_sum[tid - offset];
        }
        __syncthreads();
        if (tid >= offset) {
            s_w[tid]   += w_add;
            s_sum[tid] += sum_add;
        }
        __syncthreads();
    }

    // Write prefix sums (only for valid indices)
    if (t < OTSU_HISTOGRAM_SIZE) {
        prefix_w[t]   = s_w[tid];
        prefix_sum[t] = s_sum[tid];
    }

    // Last thread in each block writes the block total
    if (tid == blockDim.x - 1) {
        block_w_totals[blockIdx.x]   = s_w[blockDim.x - 1];
        block_sum_totals[blockIdx.x] = s_sum[blockDim.x - 1];
    }
}

// Compute variances from prefix sums
__global__ void kernel_variances(const double *prefix_w,
                                 const double *prefix_sum,
                                 const double *block_w_totals,
                                 const double *block_sum_totals,
                                 double sum_all,
                                 double *variances) {
    int t = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (t >= OTSU_HISTOGRAM_SIZE)
        return;

    // Add cumulative sum from all previous blocks
    double w_B   = prefix_w[t];
    double sum_B = prefix_sum[t];
    for (int b = 0; b < blockIdx.x; b++) {
        w_B   += block_w_totals[b];
        sum_B += block_sum_totals[b];
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
__global__ void kernel_calculate_mean(const uint16_t *image, uint64_t npixels, double *partial_sums) {
    extern __shared__ double shared_sum[];

    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    shared_sum[tid] = (idx < npixels) ? (double)image[idx] : 0.0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s *= 2) {
        if (tid < s)
            shared_sum[tid] += shared_sum[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        partial_sums[blockIdx.x] = shared_sum[0];
}

// ---------------------------------------------------------------------------
// 5.  Optimized integral-image kernels
__global__ void kernel_integral_image_row_pass(const uint16_t *image,
                                                    double *integral,
                                                    uint64_t width,
                                                    uint64_t height) {
    uint64_t y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= height)
        return;

    uint64_t row_offset = y * width;
    double row_sum = 0.0;
    for (uint64_t x = 0; x < width; x++) {
        row_sum += (double)image[row_offset + x];
        integral[row_offset + x] = row_sum;
    }
}

__global__ void kernel_integral_image_col_pass(double *integral, uint64_t width, uint64_t height) {
    uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width)
        return;

    double col_sum = 0.0;
    for (uint64_t y = 0; y < height; y++) {
        col_sum += integral[y * width + x];
        integral[y * width + x] = col_sum;
    }
}

__global__ void kernel_mean_filter_integral(const double *integral, double *temp_filtered, uint64_t width, uint64_t height, int half_window) {
    int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int64_t y1 = y - half_window;
    int64_t y2 = y + half_window;
    int64_t x1 = x - half_window;
    int64_t x2 = x + half_window;
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
__global__ void kernel_otsu_centralized_threshold(const uint16_t *image,
                                                  const double *mean_filtered,
                                                  uint8_t *output,
                                                  uint64_t npixels,
                                                  double global_mean,
                                                  double otsu_threshold) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
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

inline double* calculate_histogram_gpu(const uint16_t *d_image, uint64_t npixels) {
    // 0. allocate memory and fill with 0
    uint32_t *d_histogram;
    CHECK(cudaMalloc(&d_histogram, sizeof(uint32_t) * OTSU_HISTOGRAM_SIZE));
    CHECK(cudaMemset(d_histogram, 0, sizeof(uint32_t) * OTSU_HISTOGRAM_SIZE));

    // 1. histogram kernel launch
    int blocks = (npixels + OTSU_THREADS_PER_BLOCK - 1) / OTSU_THREADS_PER_BLOCK;
    kernel_calculate_histogram<<<blocks, OTSU_THREADS_PER_BLOCK>>>(d_image, npixels, d_histogram);

    // 2. get histogram to host memory
    uint32_t *h_histogram = new uint32_t[OTSU_HISTOGRAM_SIZE];
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_histogram, d_histogram, sizeof(uint32_t) * OTSU_HISTOGRAM_SIZE, cudaMemcpyDeviceToHost));
    // 2.1 free device memory
    CHECK(cudaFree(d_histogram));

    // 3. normalize histogram to [0,1] ( -> probability distribution)
    double *normalized = new double[OTSU_HISTOGRAM_SIZE];
    for (int i = 0; i < OTSU_HISTOGRAM_SIZE; i++)
        normalized[i] = (double)h_histogram[i] / (double)npixels;

    delete[] h_histogram;
    return normalized;
}

// return value of otsu threshold
inline uint16_t find_otsu_threshold_gpu(const double *histogram) {
    double *d_histogram;
    CHECK(cudaMalloc(&d_histogram, sizeof(double) * OTSU_HISTOGRAM_SIZE));
    CHECK(cudaMemcpy(d_histogram, histogram, sizeof(double) * OTSU_HISTOGRAM_SIZE, cudaMemcpyHostToDevice));

    // 0. Allocate prefix-scan buffers
    double *d_prefix_w, *d_prefix_sum;
    double *d_block_w_totals, *d_block_sum_totals;
    CHECK(cudaMalloc(&d_prefix_w, sizeof(double) * OTSU_HISTOGRAM_SIZE));
    CHECK(cudaMalloc(&d_prefix_sum, sizeof(double) * OTSU_HISTOGRAM_SIZE));

    int num_blocks = (OTSU_HISTOGRAM_SIZE + OTSU_THREADS_PER_BLOCK - 1) / OTSU_THREADS_PER_BLOCK;
    CHECK(cudaMalloc(&d_block_w_totals, sizeof(double) * num_blocks));
    CHECK(cudaMalloc(&d_block_sum_totals, sizeof(double) * num_blocks));

    // 0.1 expected bin value
    double sum_all = 0.0;
    for (int i = 0; i < OTSU_HISTOGRAM_SIZE; i++)
        sum_all += (double)i * histogram[i];

    size_t shared_mem = 2 * OTSU_THREADS_PER_BLOCK * sizeof(double);

    // 1. block-level prefix scan
    kernel_prefix_scan<<<num_blocks, OTSU_THREADS_PER_BLOCK, shared_mem>>>(
        d_histogram, d_prefix_w, d_prefix_sum,
        d_block_w_totals, d_block_sum_totals);
    CHECK(cudaDeviceSynchronize());

    // 2. compute variances
    double *d_variances;
    CHECK(cudaMalloc(&d_variances, sizeof(double) * OTSU_HISTOGRAM_SIZE));

    kernel_variances<<<num_blocks, OTSU_THREADS_PER_BLOCK>>>(
        d_prefix_w, d_prefix_sum, d_block_w_totals, d_block_sum_totals,
        sum_all, d_variances);
    CHECK(cudaDeviceSynchronize());

    double *h_variances = new double[OTSU_HISTOGRAM_SIZE];
    CHECK(cudaMemcpy(h_variances, d_variances, sizeof(double) * OTSU_HISTOGRAM_SIZE, cudaMemcpyDeviceToHost));

    // 3. find max variance and get the corresponding bin
    double max_variance = 0.0;
    uint16_t threshold = 0;
    for (int t = 0; t < OTSU_HISTOGRAM_SIZE; t++) {
        if (h_variances[t] > max_variance) {
            max_variance = h_variances[t];
            threshold = t;
        }
    }

    // 4. memory deallocations
    delete[] h_variances;
    CHECK(cudaFree(d_histogram));
    CHECK(cudaFree(d_variances));
    CHECK(cudaFree(d_prefix_w));
    CHECK(cudaFree(d_prefix_sum));
    CHECK(cudaFree(d_block_w_totals));
    CHECK(cudaFree(d_block_sum_totals));

    return threshold;
}

// global mean (parallel reduction)
inline double calculate_mean_gpu(const uint16_t *d_image, uint64_t npixels) {
    int blocks = (npixels + OTSU_THREADS_PER_BLOCK - 1) / OTSU_THREADS_PER_BLOCK;

    // 0. allocate gpu memory
    double *d_partial_sums;
    CHECK(cudaMalloc(&d_partial_sums, sizeof(double) * blocks));

    // 1. launch kernel
    kernel_calculate_mean<<<blocks, OTSU_THREADS_PER_BLOCK, sizeof(double) * OTSU_THREADS_PER_BLOCK>>>(
        d_image, npixels, d_partial_sums);
    CHECK(cudaDeviceSynchronize());

    // 2. get partial results to host memory
    double *h_partial = new double[blocks];
    CHECK(cudaMemcpy(h_partial, d_partial_sums, sizeof(double) * blocks, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_partial_sums));

    // 3. calculate total
    double total = 0.0;
    for (int i = 0; i < blocks; i++)
        total += h_partial[i];
    delete[] h_partial;

    return total / (double)npixels;
}

// =========================================================================
//  Host-side main function
void otsu_centralized_threshold_gpu(const uint16_t *d_image,
                                    uint8_t *d_output,
                                    uint64_t width,
                                    uint64_t height,
                                    int window_size,
                                    float th_scale) {
    uint64_t npixels = width * height;

    // 1. histogram and find otsu threshold
    double *histogram = calculate_histogram_gpu(d_image, npixels);
    double otsu_threshold = find_otsu_threshold_gpu(histogram);
    otsu_threshold *= (double)th_scale;
    delete[] histogram;

    // 2. global mean
    double global_mean = calculate_mean_gpu(d_image, npixels);

    // 3. mean filter with integral optimizations
    double *d_integral;
    CHECK(cudaMalloc(&d_integral, sizeof(double) * npixels));

    int row_blocks = (height + 255) / 256;
    kernel_integral_image_row_pass<<<row_blocks, 256>>>(
        d_image, d_integral, width, height);
    CHECK(cudaDeviceSynchronize());

    int col_blocks = (width + 255) / 256;
    kernel_integral_image_col_pass<<<col_blocks, 256>>>(
        d_integral, width, height);
    CHECK(cudaDeviceSynchronize());

    double *d_mean_filtered;
    CHECK(cudaMalloc(&d_mean_filtered, sizeof(double) * npixels));

    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    kernel_mean_filter_integral<<<grid_size, block_size>>>(
        d_integral, d_mean_filtered, width, height, (int)(window_size / 2));
    CHECK(cudaDeviceSynchronize());

    // 4. thresholding
    int blocks = (npixels + OTSU_THREADS_PER_BLOCK - 1) / OTSU_THREADS_PER_BLOCK;
    kernel_otsu_centralized_threshold<<<blocks, OTSU_THREADS_PER_BLOCK>>>(
        d_image, d_mean_filtered, d_output, npixels, global_mean, otsu_threshold);
    CHECK(cudaDeviceSynchronize());

    // 5. memory cleanup
    CHECK(cudaFree(d_integral));
    CHECK(cudaFree(d_mean_filtered));
}
