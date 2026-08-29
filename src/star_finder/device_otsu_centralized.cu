#include <stdint.h>
#include <cmath>
#include <float.h>
#include "cuda_helper.hh"
#include "common.hh"

// Constants
#define OTSU_HISTOGRAM_SIZE  65536
#define OTSU_THREADS_PER_BLOCK 1024
#define BOX_FILTER_THREADS 256
#define HIST_PRIVATES 16


// ---------------------------------------------------------------------------
// Histogram (privatized): more private copies to reduce atomic contention
// cannot use shared memory for the histogram because it is too large (4*64KB = 256KB!)
__global__ void kernel_calculate_histogram(const uint16_t *image, uint64_t npixels, uint32_t *histograms) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= npixels)
        return;

    // Each block picks a private copy to reduce cross-block contention
    uint32_t *hist = histograms + (blockIdx.x % HIST_PRIVATES) * OTSU_HISTOGRAM_SIZE;
    auto v = image[idx];
    atomicAdd(&hist[v], 1);
}

__global__ void kernel_sum_and_normalize_histograms(uint32_t *histograms, double *output, double npixels) {
    int t = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (t >= OTSU_HISTOGRAM_SIZE)
        return;

    uint32_t sum = 0;
    for (int p = 0; p < HIST_PRIVATES; p++)
        sum += histograms[p * OTSU_HISTOGRAM_SIZE + t];
    output[t] = (double)sum / npixels;
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

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            shared_sum[tid] += shared_sum[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        partial_sums[blockIdx.x] = shared_sum[0];
}

// Block-level argmax for variances
__global__ void kernel_block_max_var(const double *variances, float2 *block_results) {
    // float2.x = max variance, float2.y = bin index
    extern __shared__ float2 maxvar_shared[];
    int t = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int tid = (int)threadIdx.x;

    if (t < OTSU_HISTOGRAM_SIZE) {
        maxvar_shared[tid].x = (float)variances[t];
        maxvar_shared[tid].y = (float)t;
    } else {
        maxvar_shared[tid].x = -1.0f;
        maxvar_shared[tid].y = 0.0f;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (maxvar_shared[tid + s].x > maxvar_shared[tid].x)
                maxvar_shared[tid] = maxvar_shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        block_results[blockIdx.x] = maxvar_shared[0];
}

// ---------------------------------------------------------------------------
// Separable box filter (like CUDA samples)
__device__ void d_boxfilter_x_uint16(const uint16_t *id, double *od, int w, int r) {
    double scale = 1.0 / (double)((r << 1) + 1);
    double t;
    // left edge
    t = (double)id[0] * (double)r;
    for (int x = 0; x < (r + 1); x++)
        t += (double)id[x];
    od[0] = t * scale;
    for (int x = 1; x < (r + 1); x++) {
        t += (double)id[x + r];
        t -= (double)id[0];
        od[x] = t * scale;
    }
    // main loop
    for (int x = (r + 1); x < w - r; x++) {
        t += (double)id[x + r];
        t -= (double)id[x - r - 1];
        od[x] = t * scale;
    }
    // right edge
    for (int x = w - r; x < w; x++) {
        t += (double)id[w - 1];
        t -= (double)id[x - r - 1];
        od[x] = t * scale;
    }
}

__device__ void d_boxfilter_y(const double *id, double *od, int w, int h, int r) {
    double scale = 1.0 / (double)((r << 1) + 1);
    double t;
    // top edge
    t = id[0] * (double)r;
    for (int y = 0; y < (r + 1); y++)
        t += id[y * w];
    od[0] = t * scale;
    for (int y = 1; y < (r + 1); y++) {
        t += id[(y + r) * w];
        t -= id[0];
        od[y * w] = t * scale;
    }
    // main loop
    for (int y = (r + 1); y < (h - r); y++) {
        t += id[(y + r) * w];
        t -= id[(y - r - 1) * w];
        od[y * w] = t * scale;
    }
    // bottom edge
    for (int y = h - r; y < h; y++) {
        t += id[(h - 1) * w];
        t -= id[(y - r - 1) * w];
        od[y * w] = t * scale;
    }
}

__global__ void kernel_boxfilter_x(const uint16_t *image, double *temp, uint64_t width, uint64_t height, int r) {
    uint64_t y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= height) return;
    d_boxfilter_x_uint16(&image[y * width], &temp[y * width], (int)width, r);
}

__global__ void kernel_boxfilter_y(const double *temp, double *filtered, uint64_t width, uint64_t height, int r) {
    uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) return;
    d_boxfilter_y(&temp[x], &filtered[x], (int)width, (int)height, r);
}



// ---------------------------------------------------------------------------
// 6.  Centralized threshold kernel
__global__ void kernel_otsu_centralized_threshold(const uint16_t *image,
                                                  const double *mean_filtered,
                                                  uint8_t *output,
                                                  uint64_t npixels,
                                                  float global_mean,
                                                  float otsu_threshold) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= npixels)
        return;

    float pixel_val    = image[idx];
    float filtered_val = mean_filtered[idx];

    // T_c  =  mean_filtered  -  global_mean  +  otsu_threshold
    float pixel_threshold = filtered_val - global_mean + otsu_threshold;

    output[idx] = (pixel_val > pixel_threshold) ? 255 : 0;
}

// =========================================================================
//  Host-side helper functions (all GPU-resident, no large host transfers)

// Returns a device pointer to the normalized histogram (caller must free)
inline void calculate_histogram_gpu(const uint16_t *d_image, uint64_t npixels,
                                     double **d_out_hist_norm) {
    // 0. allocate HIST_PRIVATES private copies + 1 output copy for uint32 + 1 normalized hist
    uint32_t *d_privates;
    CHECK(cudaMalloc(&d_privates, sizeof(uint32_t) * OTSU_HISTOGRAM_SIZE * HIST_PRIVATES));
    CHECK(cudaMemset(d_privates, 0, sizeof(uint32_t) * OTSU_HISTOGRAM_SIZE * HIST_PRIVATES));

    double *d_hist_norm;
    CHECK(cudaMalloc(&d_hist_norm, sizeof(double) * OTSU_HISTOGRAM_SIZE));

    // 1. privatized histogram kernel
    int blocks = (int)((npixels + OTSU_THREADS_PER_BLOCK - 1) / OTSU_THREADS_PER_BLOCK);
    kernel_calculate_histogram<<<blocks, OTSU_THREADS_PER_BLOCK>>>(d_image, npixels, d_privates);
    CHECK(cudaDeviceSynchronize());

    // 2. sum private copies into one and normalize
    int sb = (OTSU_HISTOGRAM_SIZE + OTSU_THREADS_PER_BLOCK - 1) / OTSU_THREADS_PER_BLOCK;
    kernel_sum_and_normalize_histograms<<<sb, OTSU_THREADS_PER_BLOCK>>>(d_privates, d_hist_norm, npixels);
    CHECK(cudaDeviceSynchronize());

    // 3. free intermediates
    CHECK(cudaFree(d_privates));

    *d_out_hist_norm   = d_hist_norm;
}

// Compute Otsu thresholds entirely on GPU, return threshold bin index (0..OTSU_HISTOGRAM_SIZE-1)
inline int find_otsu_threshold_gpu(double *d_hist_norm) {
    int num_blocks = (OTSU_HISTOGRAM_SIZE + OTSU_THREADS_PER_BLOCK - 1) / OTSU_THREADS_PER_BLOCK;

    // Allocate prefix-scan buffers
    double *d_prefix_w, *d_prefix_sum;
    double *d_block_w_totals, *d_block_sum_totals;
    CHECK(cudaMalloc(&d_prefix_w,         sizeof(double) * OTSU_HISTOGRAM_SIZE));
    CHECK(cudaMalloc(&d_prefix_sum,       sizeof(double) * OTSU_HISTOGRAM_SIZE));
    CHECK(cudaMalloc(&d_block_w_totals,   sizeof(double) * num_blocks));
    CHECK(cudaMalloc(&d_block_sum_totals, sizeof(double) * num_blocks));

    size_t shared_mem = 2 * OTSU_THREADS_PER_BLOCK * sizeof(double);

    // 1. block-level prefix scan
    kernel_prefix_scan<<<num_blocks, OTSU_THREADS_PER_BLOCK, shared_mem>>>(
        d_hist_norm, d_prefix_w, d_prefix_sum,
        d_block_w_totals, d_block_sum_totals);
    CHECK(cudaDeviceSynchronize());

    // Derive sum_all from block_sum_totals (sum of all block totals = total sum)
    double *h_block_sum = new double[num_blocks];
    CHECK(cudaMemcpy(h_block_sum, d_block_sum_totals, sizeof(double) * num_blocks, cudaMemcpyDeviceToHost));
    double sum_all = 0.0;
    for (int b = 0; b < num_blocks; b++) sum_all += h_block_sum[b];
    delete[] h_block_sum;

    // 2. compute variances
    double *d_variances;
    CHECK(cudaMalloc(&d_variances, sizeof(double) * OTSU_HISTOGRAM_SIZE));

    kernel_variances<<<num_blocks, OTSU_THREADS_PER_BLOCK>>>(
        d_prefix_w, d_prefix_sum, d_block_w_totals, d_block_sum_totals,
        sum_all, d_variances);
    CHECK(cudaDeviceSynchronize());

    // 3. block-level argmax (reduces to num_blocks candidates)
    float2 *d_block_results;
    CHECK(cudaMalloc(&d_block_results, sizeof(float2) * num_blocks));

    kernel_block_max_var<<<num_blocks, OTSU_THREADS_PER_BLOCK, sizeof(float2) * OTSU_THREADS_PER_BLOCK>>>(
        d_variances, d_block_results);
    CHECK(cudaDeviceSynchronize());

    // 4. copy block results to host and find final max (only OTSU_THREADS_PER_BLOCK values)
    float2 *h_block_results = new float2[num_blocks];
    CHECK(cudaMemcpy(h_block_results, d_block_results, sizeof(float2) * num_blocks, cudaMemcpyDeviceToHost));

    int threshold_bin = 0;
    double max_variance = -1.0;
    for (int i = 0; i < num_blocks; i++) {
        if ((double)h_block_results[i].x > max_variance) {
            max_variance = (double)h_block_results[i].x;
            threshold_bin = (int)h_block_results[i].y;
        }
    }

    delete[] h_block_results;

    // 5. cleanup
    CHECK(cudaFree(d_prefix_w));
    CHECK(cudaFree(d_prefix_sum));
    CHECK(cudaFree(d_block_w_totals));
    CHECK(cudaFree(d_block_sum_totals));
    CHECK(cudaFree(d_variances));
    CHECK(cudaFree(d_block_results));

    return threshold_bin;
}

// Global mean (parallel reduction on GPU, tiny host transfer)
inline double calculate_mean_gpu(const uint16_t *d_image, uint64_t npixels) {
    int blocks = (int)((npixels + OTSU_THREADS_PER_BLOCK - 1) / OTSU_THREADS_PER_BLOCK);

    double *d_partial_sums;
    CHECK(cudaMalloc(&d_partial_sums, sizeof(double) * blocks));

    kernel_calculate_mean<<<blocks, OTSU_THREADS_PER_BLOCK, sizeof(double) * OTSU_THREADS_PER_BLOCK>>>(
        d_image, npixels, d_partial_sums);
    CHECK(cudaDeviceSynchronize());

    // Small host transfer (blocks ~= npixels/256, e.g. 256K for 67MP)
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
void otsu_centralized_threshold_gpu(const uint16_t *d_image,
                                    uint8_t *d_output,
                                    uint64_t width,
                                    uint64_t height,
                                    int window_size,
                                    float th_scale) {
    uint64_t npixels = width * height;

    // 1. GPU-resident histogram + normalization
    double *d_hist_norm;
    calculate_histogram_gpu(d_image, npixels, &d_hist_norm);

    // 2. find otsu threshold (entirely on GPU, returns single bin index)
    int otsu_bin = find_otsu_threshold_gpu(d_hist_norm);
    double otsu_threshold = (double)otsu_bin;
    otsu_threshold *= (double)th_scale;

    // 2.1 free histogram memory
    CHECK(cudaFree(d_hist_norm));

    // 3. global mean
    double global_mean = calculate_mean_gpu(d_image, npixels);

    // 4. mean filter with separable box filter (like CUDA samples)
    double *d_mean_filtered;
    CHECK(cudaMalloc(&d_mean_filtered, sizeof(double) * npixels));

    double *d_temp;
    CHECK(cudaMalloc(&d_temp, sizeof(double) * npixels));

    int r = (int)(window_size / 2);
    int row_blocks = (int)((height + BOX_FILTER_THREADS - 1) / BOX_FILTER_THREADS);
    kernel_boxfilter_x<<<row_blocks, BOX_FILTER_THREADS>>>(d_image, d_temp, width, height, r);
    CHECK(cudaDeviceSynchronize());

    int col_blocks = (int)((width + BOX_FILTER_THREADS - 1) / BOX_FILTER_THREADS);
    kernel_boxfilter_y<<<col_blocks, BOX_FILTER_THREADS>>>(d_temp, d_mean_filtered, width, height, r);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaFree(d_temp));

    // 5. thresholding
    int blocks = (npixels + OTSU_THREADS_PER_BLOCK - 1) / OTSU_THREADS_PER_BLOCK;
    kernel_otsu_centralized_threshold<<<blocks, OTSU_THREADS_PER_BLOCK>>>(
        d_image, d_mean_filtered, d_output, npixels, global_mean, otsu_threshold);
    CHECK(cudaDeviceSynchronize());

    // 5.1 memory cleanup
    CHECK(cudaFree(d_mean_filtered));
}
