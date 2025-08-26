#ifndef CUDA_DEVICE_DEBAYER_H
#define CUDA_DEVICE_DEBAYER_H

__device__ inline long clamp(long v, long lo, long hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__global__ void demosaic_bilinear_rggb_kernel( const u_int16_t *gray_all, u_int16_t *rgb_all, long width, long height, u_int16_t image_count){
    u_int64_t idx_global = blockIdx.x * blockDim.x + threadIdx.x;
    u_int64_t npixels = width * height;
    u_int64_t total_pixels = npixels * image_count;

    if (idx_global >= total_pixels) return;

    // indice immagine e pixel
    u_int16_t image_idx = idx_global / npixels;
    u_int64_t pixel_idx = idx_global % npixels;

    long y = pixel_idx / width;
    long x = pixel_idx % width;

    const u_int16_t *gray = gray_all + image_idx*npixels;

    // coord clampate
    long xm1 = clamp(x-1, 0, width-1);
    long xp1 = clamp(x+1, 0, width-1);
    long ym1 = clamp(y-1, 0, height-1);
    long yp1 = clamp(y+1, 0, height-1);

    // valori vicini
    u_int16_t c      = gray[y*width + x];     // centro
    u_int16_t l      = gray[y*width + xm1];   // left
    u_int16_t r      = gray[y*width + xp1];   // right
    u_int16_t u      = gray[ym1*width + x];   // up
    u_int16_t d      = gray[yp1*width + x];   // down
    u_int16_t ul     = gray[ym1*width + xm1];
    u_int16_t ur     = gray[ym1*width + xp1];
    u_int16_t dl     = gray[yp1*width + xm1];
    u_int16_t dr     = gray[yp1*width + xp1];

    u_int16_t R=0, G=0, B=0;

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
    u_int16_t *rgb = rgb_all + image_idx * npixels * 3;
    rgb[pixel_idx]            = R;
    rgb[pixel_idx + npixels]  = G;
    rgb[pixel_idx + 2*npixels]= B;
}

#endif