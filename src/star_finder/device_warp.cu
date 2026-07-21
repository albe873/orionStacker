#include "star_finder.h"
#include "cuda_helper.h"

// ---------------------------------------------------------------------------
// GPU-only affine warp for planar u16 RGB data (R, G, B stored as contiguous
// planes: [R: npixels | G: npixels | B: npixels]).
//
// Bilinear interpolation with per-tap constant-zero border: each of the four
// sampling taps that falls outside the source image contributes 0 while the
// valid taps keep their weights.  This matches the semantics of
//   cv::warpAffine(..., cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0))
// Works on CUDA managed memory without ever migrating to the CPU.
// ---------------------------------------------------------------------------

// Fast path: the full 2x2 neighbourhood is inside the image (no tap guards).
__device__ inline u_int16_t bilinear_interior(
    const u_int16_t *__restrict__ plane,
    int sx, int sy, float fx, float fy, int width)
{
    const int o = sy * width + sx;

    const float v00 = (float)plane[o];
    const float v10 = (float)plane[o + 1];
    const float v01 = (float)plane[o + width];
    const float v11 = (float)plane[o + width + 1];

    // fmaf keeps full float precision through the two lerp stages.
    const float v0 = fmaf(fx, v10 - v00, v00);
    const float v1 = fmaf(fx, v11 - v01, v01);
    const float v  = fmaf(fy, v1 - v0, v0);

    // __float2int_rn = round-to-nearest-even (same rounding as cvRound),
    // then saturate defensively to the u16 range.
    return (u_int16_t)min(max(__float2int_rn(v), 0), 65535);
}

// Border path: guard every tap individually (BORDER_CONSTANT(0) semantics).
__device__ inline u_int16_t bilinear_border(
    const u_int16_t *__restrict__ plane,
    int sx, int sy, float fx, float fy, int width, int height)
{
    const bool x0_ok = (sx     >= 0) && (sx     < width);
    const bool x1_ok = (sx + 1 >= 0) && (sx + 1 < width);
    const bool y0_ok = (sy     >= 0) && (sy     < height);
    const bool y1_ok = (sy + 1 >= 0) && (sy + 1 < height);

    const float v00 = (x0_ok && y0_ok) ? (float)plane[sy * width + sx]           : 0.0f;
    const float v10 = (x1_ok && y0_ok) ? (float)plane[sy * width + sx + 1]       : 0.0f;
    const float v01 = (x0_ok && y1_ok) ? (float)plane[(sy + 1) * width + sx]     : 0.0f;
    const float v11 = (x1_ok && y1_ok) ? (float)plane[(sy + 1) * width + sx + 1] : 0.0f;

    const float v0 = fmaf(fx, v10 - v00, v00);
    const float v1 = fmaf(fx, v11 - v01, v01);
    const float v  = fmaf(fy, v1 - v0, v0);

    return (u_int16_t)min(max(__float2int_rn(v), 0), 65535);
}

// ---------------------------------------------------------------------------
// Optimized warp kernel with vectorized loads/stores (uint2 = 2 pixels = 4 bytes)
// and shared memory tiling for bilinear interpolation.
// Processes 2 pixels per thread in X dimension.
// ---------------------------------------------------------------------------
#define TILE_W 32
#define TILE_H 16
#define TILE_W_LOAD (TILE_W + 1)  // +1 for bilinear right neighbor
#define TILE_H_LOAD (TILE_H + 1)  // +1 for bilinear bottom neighbor

__global__ void warp_affine_planar_kernel_opt(
    const u_int16_t *__restrict__ src,
    u_int16_t *__restrict__ dst,
    double a, double b, double tx,
    double c, double d, double ty,
    int width, int height)
{
    // Shared memory tile for one plane: (TILE_H+1) x (TILE_W+1) elements
    // We process 3 planes sequentially to limit shared memory usage
    extern __shared__ u_int16_t smem_plane[];

    const int npixels = width * height;
    const int plane_stride = npixels;

    // Each thread processes 2 adjacent pixels in X
    const int x_base = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y >= height) return;

    // Precompute affine transform for the two output pixels
    const float x0_f = (float)x_base;
    const float x1_f = (float)(x_base + 1);
    const float y_f  = (float)y;

    const float src_x0 = (float)(a * (double)x0_f + b * (double)y_f + tx);
    const float src_y0 = (float)(c * (double)x0_f + d * (double)y_f + ty);
    const float src_x1 = (float)(a * (double)x1_f + b * (double)y_f + tx);
    const float src_y1 = (float)(c * (double)x1_f + d * (double)y_f + ty);

    // Bilinear coordinates for pixel 0
    const int sx0 = (int)floorf(src_x0);
    const int sy0 = (int)floorf(src_y0);
    const float fx0 = src_x0 - (float)sx0;
    const float fy0 = src_y0 - (float)sy0;

    // Bilinear coordinates for pixel 1
    const int sx1 = (int)floorf(src_x1);
    const int sy1 = (int)floorf(src_y1);
    const float fx1 = src_x1 - (float)sx1;
    const float fy1 = src_y1 - (float)sy1;

    // Check if both pixels are in bounds (for early exit)
    const bool pixel0_valid = (x_base < width);
    const bool pixel1_valid = (x_base + 1 < width);

    if (!pixel0_valid && !pixel1_valid) return;

    // Process each plane (R, G, B) sequentially to reuse shared memory
    for (int plane = 0; plane < 3; ++plane) {
        const u_int16_t *src_plane = src + plane * plane_stride;
        u_int16_t *dst_plane = dst + plane * plane_stride;

        // Determine the source tile bounds needed for this thread block
        // We need to load a tile covering [min(sx0,sx1) ... max(sx0,sx1)+1] x [min(sy0,sy1) ... max(sy0,sy1)+1]
        // For simplicity and to handle all cases, load the full tile for the block
        const int tile_sx_min = blockIdx.x * blockDim.x * 2 - 1;
        const int tile_sy_min = blockIdx.y * blockDim.y - 1;
        const int tile_sx_max = tile_sx_min + TILE_W_LOAD;
        const int tile_sy_max = tile_sy_min + TILE_H_LOAD;

        // Cooperative load of source tile into shared memory
        for (int load_y = threadIdx.y; load_y < TILE_H_LOAD; load_y += blockDim.y) {
            const int src_y = tile_sy_min + load_y;
            for (int load_x = threadIdx.x; load_x < TILE_W_LOAD; load_x += blockDim.x) {
                const int src_x = tile_sx_min + load_x;
                u_int16_t val = 0;
                if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height) {
                    val = src_plane[src_y * width + src_x];
                }
                smem_plane[load_y * TILE_W_LOAD + load_x] = val;
            }
        }
        __syncthreads();

        // Now compute bilinear interpolation from shared memory
        u_int16_t result0 = 0, result1 = 0;

        if (pixel0_valid) {
            // Map source coordinates to shared memory indices
            const int smem_sx0 = sx0 - tile_sx_min;
            const int smem_sy0 = sy0 - tile_sy_min;

            if (smem_sx0 >= 0 && smem_sx0 < TILE_W_LOAD - 1 &&
                smem_sy0 >= 0 && smem_sy0 < TILE_H_LOAD - 1) {
                // Fast path: all 4 taps in shared memory
                const int idx00 = smem_sy0 * TILE_W_LOAD + smem_sx0;
                const float v00 = (float)smem_plane[idx00];
                const float v10 = (float)smem_plane[idx00 + 1];
                const float v01 = (float)smem_plane[idx00 + TILE_W_LOAD];
                const float v11 = (float)smem_plane[idx00 + TILE_W_LOAD + 1];

                const float v0 = fmaf(fx0, v10 - v00, v00);
                const float v1 = fmaf(fx0, v11 - v01, v01);
                const float v  = fmaf(fy0, v1 - v0, v0);
                result0 = (u_int16_t)min(max(__float2int_rn(v), 0), 65535);
            } else {
                // Fallback to global memory for border pixels
                result0 = bilinear_border(src_plane, sx0, sy0, fx0, fy0, width, height);
            }
        }

        if (pixel1_valid) {
            const int smem_sx1 = sx1 - tile_sx_min;
            const int smem_sy1 = sy1 - tile_sy_min;

            if (smem_sx1 >= 0 && smem_sx1 < TILE_W_LOAD - 1 &&
                smem_sy1 >= 0 && smem_sy1 < TILE_H_LOAD - 1) {
                const int idx00 = smem_sy1 * TILE_W_LOAD + smem_sx1;
                const float v00 = (float)smem_plane[idx00];
                const float v10 = (float)smem_plane[idx00 + 1];
                const float v01 = (float)smem_plane[idx00 + TILE_W_LOAD];
                const float v11 = (float)smem_plane[idx00 + TILE_W_LOAD + 1];

                const float v0 = fmaf(fx1, v10 - v00, v00);
                const float v1 = fmaf(fx1, v11 - v01, v01);
                const float v  = fmaf(fy1, v1 - v0, v0);
                result1 = (u_int16_t)min(max(__float2int_rn(v), 0), 65535);
            } else {
                result1 = bilinear_border(src_plane, sx1, sy1, fx1, fy1, width, height);
            }
        }

        // Write results
        const int dst_idx = y * width + x_base;
        if (pixel0_valid) {
            dst_plane[dst_idx] = result0;
        }
        if (pixel1_valid) {
            dst_plane[dst_idx + 1] = result1;
        }

        __syncthreads();  // Ensure shared memory not overwritten for next plane
    }
}

// Original kernel kept for compatibility
__global__ void warp_affine_planar_kernel(
    const u_int16_t *__restrict__ src,
    u_int16_t *__restrict__ dst,
    double a, double b, double tx,
    double c, double d, double ty,
    int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Inverse mapping: for each output pixel (x,y) in the aligned image,
    // find the corresponding location in the source image.
    //   dst(x,y) = src(M_{00}*x + M_{01}*y + M_{02}, M_{10}*x + M_{11}*y + M_{12})
    // The affine is evaluated in double for accuracy, then cast to float once.
    const float src_x = (float)(a * (double)x + b * (double)y + tx);
    const float src_y = (float)(c * (double)x + d * (double)y + ty);

    // Bilinear geometry computed once and shared by all three planes.
    // floorf (not an int cast) so negative coordinates floor correctly
    // and the fractional weights stay in [0,1).
    const int   sx = (int)floorf(src_x);
    const int   sy = (int)floorf(src_y);
    const float fx = src_x - (float)sx;
    const float fy = src_y - (float)sy;

    const long npixels = (long)width * (long)height;
    const int  idx     = y * width + x;

    const bool interior = (sx >= 0) && (sy >= 0) &&
                          (sx + 1 < width) && (sy + 1 < height);

    if (interior) {
        dst[idx]               = bilinear_interior(src,               sx, sy, fx, fy, width);
        dst[idx + npixels]     = bilinear_interior(src + npixels,     sx, sy, fx, fy, width);
        dst[idx + 2 * npixels] = bilinear_interior(src + 2 * npixels, sx, sy, fx, fy, width);
    } else {
        dst[idx]               = bilinear_border(src,               sx, sy, fx, fy, width, height);
        dst[idx + npixels]     = bilinear_border(src + npixels,     sx, sy, fx, fy, width, height);
        dst[idx + 2 * npixels] = bilinear_border(src + 2 * npixels, sx, sy, fx, fy, width, height);
    }
}

// ---------------------------------------------------------------------------
// GPU path – launches the optimized kernel with vectorization and shared memory.
// ---------------------------------------------------------------------------
void warp_affine_planar_gpu(const u_int16_t *source, u_int16_t *dest,
                            const cv::Mat &affine_2x3, long width, long height) {
    // OpenCV returns CV_64F (double).
    // cv::warpAffine (without WARP_INVERSE_MAP) applies M^{-1} to each output
    // pixel to find the source location:
    //   dst(x,y) = src( M^{-1} * (x,y,1)^T )
    //
    // The affine matrix M = [a b tx; c d ty] maps source->destination.
    // We need its inverse to match the CPU path:
    //   M^{-1} = [ d/det  -b/det  (b*ty - d*tx)/det ]
    //            [ -c/det  a/det  (c*tx - a*ty)/det ]
    // where det = a*d - b*c.
    double a  = affine_2x3.at<double>(0, 0);
    double b  = affine_2x3.at<double>(0, 1);
    double tx = affine_2x3.at<double>(0, 2);
    double c  = affine_2x3.at<double>(1, 0);
    double d  = affine_2x3.at<double>(1, 1);
    double ty = affine_2x3.at<double>(1, 2);

    double det = a * d - b * c;
    double inv_a  =  d / det;
    double inv_b  = -b / det;
    double inv_tx = (b * ty - d * tx) / det;
    double inv_c  = -c / det;
    double inv_d  =  a / det;
    double inv_ty = (c * tx - a * ty) / det;

    // Optimized kernel: 32x16 block, 2 pixels per thread in X
    dim3 block(16, 16);  // 16*16 = 256 threads, each processes 2 pixels = 32x16 tile
    dim3 grid((width + 31) / 32,  // 32 pixels per block in X (2 per thread * 16 threads)
              (height + 15) / 16);

    // Shared memory per block: (TILE_H+1) * (TILE_W+1) * sizeof(u_int16_t)
    // TILE_W = 32, TILE_H = 16, so (17 * 33) = 561 elements = 1122 bytes per plane
    size_t shared_mem_per_plane = (16 + 1) * (32 + 1) * sizeof(u_int16_t);

    warp_affine_planar_kernel_opt<<<grid, block, shared_mem_per_plane>>>(
        source, dest,
        inv_a, inv_b, inv_tx,
        inv_c, inv_d, inv_ty,
        (int)width, (int)height);

    CHECK(cudaDeviceSynchronize());
}