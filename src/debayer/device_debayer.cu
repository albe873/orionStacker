#include "cuda_helper.hh"
#include "debayer.hh"

#include "MHC_filters.cuh"
#include "MHC_apply.cuh"


__device__ inline long clamp(long v, long lo, long hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ inline uint16_t clamp_u16(float v) {
    if (v < 0.0f) return 0;
    if (v > 65535.0f) return 65535;
    return (uint16_t)(v + 0.5f); // round-to-nearest
}

__global__ void demosaic_bilinear_rggb_kernel( const uint16_t* __restrict__ gray_all, uint16_t* __restrict__ rgb_all, long width, long height, uint16_t image_count){
    uint64_t idx_global = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t npixels = width * height;
    uint64_t total_pixels = npixels * image_count;

    if (idx_global >= total_pixels) return;

    // indice immagine e pixel
    uint16_t image_idx = idx_global / npixels;
    uint64_t pixel_idx = idx_global % npixels;

    long y = pixel_idx / width;
    long x = pixel_idx % width;

    const uint16_t *gray = gray_all + image_idx*npixels;

    // coord clampate
    long xm1 = clamp(x-1, 0, width-1);
    long xp1 = clamp(x+1, 0, width-1);
    long ym1 = clamp(y-1, 0, height-1);
    long yp1 = clamp(y+1, 0, height-1);

    // valori vicini
    uint16_t c      = gray[y*width + x];     // centro
    uint16_t l      = gray[y*width + xm1];   // left
    uint16_t r      = gray[y*width + xp1];   // right
    uint16_t u      = gray[ym1*width + x];   // up
    uint16_t d      = gray[yp1*width + x];   // down
    uint16_t ul     = gray[ym1*width + xm1];
    uint16_t ur     = gray[ym1*width + xp1];
    uint16_t dl     = gray[yp1*width + xm1];
    uint16_t dr     = gray[yp1*width + xp1];

    uint16_t R=0, G=0, B=0;

    if ((y % 2 == 0) && (x % 2 == 0)) {           // pixel R
        R = c;
        G = (l + r + u + d) / 4;
        B = (ul + ur + dl + dr) / 4;
    } else if ((y % 2 == 0) && (x % 2 == 1)) {    // pixel G su riga R
        R = (l + r) / 2;
        G = c;
        B = (u + d) / 2;
    } else if ((y % 2 == 1) && (x % 2 == 0)) {    // pixel G su riga B
        R = (u + d) / 2;
        G = c;
        B = (l + r) / 2;
    } else {                                      // pixel B
        R = (ul + ur + dl + dr) / 4;
        G = (l + r + u + d) / 4;
        B = c;
    }

    // scrittura planare
    uint16_t *rgb = rgb_all + image_idx * npixels * 3;
    rgb[pixel_idx]            = R;
    rgb[pixel_idx + npixels]  = G;
    rgb[pixel_idx + 2*npixels]= B;
}

__global__ void demosaic_mhc_rggb_kernel( const u_int16_t * __restrict__ gray_all, u_int16_t * __restrict__ rgb_all, long width, long height, u_int16_t image_count){
    // indice globale pixel su TUTTE le immagini
    uint64_t idx_global = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t npixels = (uint64_t)width * (uint64_t)height;
    uint64_t total_pixels = npixels * (uint64_t)image_count;

    if (idx_global >= total_pixels)
        return;

    // immagine e pixel
    uint16_t image_idx = (uint16_t)(idx_global / npixels);
    uint64_t pixel_idx = idx_global % npixels;

    int y = (int)(pixel_idx / width);
    int x = (int)(pixel_idx % width);

    const uint16_t *gray = gray_all + (uint64_t)image_idx * npixels;
    uint16_t *rgb        = rgb_all  + (uint64_t)image_idx * (npixels * 3ull);

    // valore noto da CFA
    uint16_t c_u16 = gray[y * (int)width + x];
    float c = (float)c_u16;

    float Rf=0.f, Gf=0.f, Bf=0.f;

    // Schema RGGB:
    // (y%2==0, x%2==0) -> R
    // (y%2==0, x%2==1) -> G (riga R)
    // (y%2==1, x%2==0) -> G (riga B)
    // (y%2==1, x%2==1) -> B
    bool y_even = (y & 1) == 0;
    bool x_even = (x & 1) == 0;

    if (y_even && x_even) {
        // ----- Pixel R -----
        Rf = c;
        Gf = apply_kernel_5x5(gray, (int)width, (int)height, x, y, KERNEL_G_at_RB);
        Bf = apply_kernel_5x5(gray, (int)width, (int)height, x, y, KERNEL_RB_at_opposite);
    }
    else if (y_even && !x_even) {
        // ----- Pixel G su riga R -----
        Gf = c;
        // R a green (red rows): kernel "diag"
        Rf = apply_kernel_5x5(gray, (int)width, (int)height, x, y, KERNEL_RB_at_G_diag);
        // B a green (red rows): kernel trasposto "cross"
        Bf = apply_kernel_5x5(gray, (int)width, (int)height, x, y, KERNEL_RB_at_G_cross);
    }
    else if (!y_even && x_even) {
        // ----- Pixel G su riga B -----
        Gf = c;
        // B a green (blue rows): kernel "diag"
        Bf = apply_kernel_5x5(gray, (int)width, (int)height, x, y, KERNEL_RB_at_G_diag);
        // R a green (blue rows): kernel trasposto "cross"
        Rf = apply_kernel_5x5(gray, (int)width, (int)height, x, y, KERNEL_RB_at_G_cross);
    }
    else { // (!y_even && !x_even)
        // ----- Pixel B -----
        Bf = c;
        Gf = apply_kernel_5x5(gray, (int)width, (int)height, x, y, KERNEL_G_at_RB);
        Rf = apply_kernel_5x5(gray, (int)width, (int)height, x, y, KERNEL_RB_at_opposite);
    }

    // clamp e scrittura planare
    rgb[pixel_idx]             = clamp_u16(Rf);
    rgb[pixel_idx + npixels]   = clamp_u16(Gf);
    rgb[pixel_idx + 2*npixels] = clamp_u16(Bf);
}


// ------------ wrapper functions ------------

void demosaic_bilinear_rggb_gpu(const uint16_t* __restrict__ gray_all, uint16_t * __restrict__ rgb_all,
                                long width, long height, uint16_t image_count) {
    uint64_t npixels = width*height;
    dim3 block_size(512);
    dim3 grid_size((npixels*image_count + block_size.x - 1)/block_size.x);
    demosaic_bilinear_rggb_kernel<<<grid_size, block_size>>>(gray_all, rgb_all, width, height, image_count);
    CHECK(cudaDeviceSynchronize());
}

void demosaic_mhc_rggb_gpu(const uint16_t* __restrict__ gray_all, uint16_t * __restrict__ rgb_all,
                           long width, long height, uint16_t image_count) {
    uint64_t npixels = (uint64_t)width * (uint64_t)height;
    dim3 block_size(512);
    dim3 grid_size((npixels*image_count + block_size.x - 1)/block_size.x);
    demosaic_mhc_rggb_kernel<<<grid_size, block_size>>>(gray_all, rgb_all, width, height, image_count);
    CHECK(cudaDeviceSynchronize());
}