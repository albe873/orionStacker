#include "star_finder.hh"
#include "cuda_helper.hh"


// full 2x2 neighbourhood is inside the image
__device__ inline uint16_t bilinear_interior(
    const uint16_t *__restrict__ plane,
    int64_t sx, int64_t sy,
    float fx, float fy, int64_t width) {
    int64_t base = sy * width + sx;

    float v00 = (float)plane[base];
    float v10 = (float)plane[base + 1];
    float v01 = (float)plane[base + width];       // needs to get to the lower line
    float v11 = (float)plane[base + width + 1];

    // fmaf keeps full float precision through the two lerp stages.
    float v0 = fmaf(fx, v10 - v00, v00);
    float v1 = fmaf(fx, v11 - v01, v01);
    float v  = fmaf(fy, v1 - v0, v0);

    // __float2int_rn = round-to-nearest-even (same rounding as cvRound)
    // then saturate defensively to the u16 range.
    return (uint16_t)min(max(__float2int_rn(v), 0), 65535);
}

__device__ inline uint16_t bilinear_border(
    const uint16_t *__restrict__ plane,
    int64_t sx, int64_t sy, 
    float fx, float fy,
    int64_t width, int64_t height)
{
    // fast check
    if (sx + 1 < 0 || sx > width ||
        sy + 1 < 0 || sy > height)
        return 0;

    // Clamp each of the 4 bilinear taps individually.
    // Out-of-bounds become 0 (BORDER_CONSTANT)
    float v00 = (sx >= 0 && sx < width && sy >= 0 && sy < height)
                ? plane[sy * width + sx] : 0.0f;
    float v10 = (sx + 1 >= 0 && sx + 1 < width && sy >= 0 && sy < height)
                ? plane[sy * width + (sx + 1)] : 0.0f;
    float v01 = (sx >= 0 && sx < width && sy + 1 >= 0 && sy + 1 < height)
                ? plane[(sy + 1) * width + sx] : 0.0f;
    float v11 = (sx + 1 >= 0 && sx + 1 < width && sy + 1 >= 0 && sy + 1 < height)
                ? plane[(sy + 1) * width + (sx + 1)] : 0.0f; 

    float v0 = fmaf(fx, v10 - v00, v00);
    float v1 = fmaf(fx, v11 - v01, v01);
    float v  = fmaf(fy, v1 - v0, v0);

    return (uint16_t)min(max(__float2int_rn(v), 0), 65535);
}

__global__ void warp_affine_planar_kernel(
    const uint16_t *__restrict__ src,
    uint16_t *__restrict__ dst,
    double a, double b, double tx,
    double c, double d, double ty,
    int64_t width, int64_t height)
{
    int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // Inverse mapping: for each output pixel (x,y) in the aligned image,
    // find the corresponding location in the source image.
    //   dst(x,y) = src(M_{00}*x + M_{01}*y + M_{02}, M_{10}*x + M_{11}*y + M_{12})
    // The affine is evaluated in double for accuracy, then cast to float once.
    float src_x = a*x + b*y + tx;
    float src_y = c*x + d*y + ty;

    // Bilinear geometry computed once and shared by all three planes.
    // floorf (not an int cast) so negative coordinates floor correctly
    // and the fractional weights stay in [0,1).
    int64_t sx = floorf(src_x);
    int64_t sy = floorf(src_y);
    float fx = src_x - (float)sx;
    float fy = src_y - (float)sy;

    int64_t npixels = width * height;
    int64_t idx     = y * width + x;

    bool interior = (sx >= 0) && (sy >= 0) &&
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
void warp_affine_planar_gpu(const uint16_t *source, uint16_t *dest,
                            const cv::Mat &affine_2x3, int64_t width, int64_t height) {
    // OpenCV returns CV_64F (double).
    // cv::warpAffine applies M^-1 to each output pixel to find the source location value:
    //   dst(x,y) = src( M^-1 * (x,y,1)^T )
    //
    // The affine matrix M = [a b tx; c d ty] maps source->destination.
    //   M^-1   = [ d/det  -b/det  (b*ty - d*tx)/det ]
    //            [ -c/det  a/det  (c*tx - a*ty)/det ]
    // with det = a*d - b*c.
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

    dim3 block(16, 16);
    dim3 grid((width  + block.x - 1) / block.x, 
              (height + block.y - 1) / block.y);

    warp_affine_planar_kernel<<<grid, block>>>(
        source, dest,
        inv_a, inv_b, inv_tx,
        inv_c, inv_d, inv_ty,
        width, height
    );
    
    CHECK(cudaDeviceSynchronize());
}