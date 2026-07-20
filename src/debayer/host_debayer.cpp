#include "debayer.h"
#include "MHC_filters.h"
#include <cstdint>
#include <algorithm>

// ============================================================================
// Helper functions
// ============================================================================

inline long clamp_index(long v, long lo, long hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

inline uint16_t clamp_u16(float v) {
    if (v < 0.0f) return 0;
    if (v > 65535.0f) return 65535;
    return static_cast<uint16_t>(v + 0.5f); // round-to-nearest
}

// ============================================================================
// CPU: Apply 5x5 kernel (MHC)
// ============================================================================
static inline float apply_kernel_5x5_cpu(
    const uint16_t* __restrict__ gray,
    long width,
    long height,
    long x,
    long y,
    const float kernel[5][5]
) {
    float acc = 0.0f;

    #pragma unroll
    for (int ky = -2; ky <= 2; ky++) {
        long yy = clamp_index(y + ky, 0, height - 1);

        #pragma unroll
        for (int kx = -2; kx <= 2; kx++) {
            long xx = clamp_index(x + kx, 0, width - 1);

            float coeff = kernel[ky + 2][kx + 2];
            float val   = static_cast<float>(gray[yy * width + xx]);

            acc += coeff * val;
        }
    }

    // Normalizzazione: tutti i kernel Malvar sono divisi per 8
    return acc / 8.0f;
}

// ============================================================================
// CPU: Bilinear debayering (RGGB pattern)
// ============================================================================
void demosaic_bilinear_rggb_cpu(
    const uint16_t* __restrict__ gray_all,
    uint16_t* __restrict__ rgb_all,
    long width, long height,
    uint16_t image_count
) {
    const long npixels = width * height;
    const long total_pixels = npixels * image_count;

    for (uint16_t img_idx = 0; img_idx < image_count; ++img_idx) {
        const uint16_t *gray = gray_all + img_idx * npixels;
        uint16_t *rgb = rgb_all + img_idx * npixels * 3;

        #pragma omp parallel for
        for (long y = 0; y < height; ++y) {
            for (long x = 0; x < width; ++x) {
                long pixel_idx = y * width + x;

                // Coordinate clampate
                long xm1 = clamp_index(x - 1, 0, width - 1);
                long xp1 = clamp_index(x + 1, 0, width - 1);
                long ym1 = clamp_index(y - 1, 0, height - 1);
                long yp1 = clamp_index(y + 1, 0, height - 1);

                // Valori vicini
                uint16_t c  = gray[y * width + x];
                uint16_t l  = gray[y * width + xm1];
                uint16_t r  = gray[y * width + xp1];
                uint16_t u  = gray[ym1 * width + x];
                uint16_t d  = gray[yp1 * width + x];
                uint16_t ul = gray[ym1 * width + xm1];
                uint16_t ur = gray[ym1 * width + xp1];
                uint16_t dl = gray[yp1 * width + xm1];
                uint16_t dr = gray[yp1 * width + xp1];

                uint16_t R = 0, G = 0, B = 0;

                // Schema RGGB:
                // (y%2==0, x%2==0) -> R
                // (y%2==0, x%2==1) -> G (riga R)
                // (y%2==1, x%2==0) -> G (riga B)
                // (y%2==1, x%2==1) -> B
                bool y_even = (y % 2 == 0);
                bool x_even = (x % 2 == 0);

                if (y_even && x_even) {
                    // Pixel R
                    R = c;
                    G = (l + r + u + d) / 4;
                    B = (ul + ur + dl + dr) / 4;
                } else if (y_even && !x_even) {
                    // Pixel G su riga R
                    R = (l + r) / 2;
                    G = c;
                    B = (u + d) / 2;
                } else if (!y_even && x_even) {
                    // Pixel G su riga B
                    R = (u + d) / 2;
                    G = c;
                    B = (l + r) / 2;
                } else {
                    // Pixel B
                    R = (ul + ur + dl + dr) / 4;
                    G = (l + r + u + d) / 4;
                    B = c;
                }

                // Scrittura planare (R, G, B piani separati)
                rgb[pixel_idx]             = R;
                rgb[pixel_idx + npixels]   = G;
                rgb[pixel_idx + 2*npixels] = B;
            }
        }
    }
}

// ============================================================================
// CPU: Malvar-He-Cutler (MHC) debayering (RGGB pattern)
// ============================================================================
void demosaic_mhc_rggb_cpu(
    const uint16_t* __restrict__ gray_all,
    uint16_t* __restrict__ rgb_all,
    long width,
    long height,
    uint16_t image_count
) {
    const long npixels = width * height;

    for (uint16_t img_idx = 0; img_idx < image_count; ++img_idx) {
        const uint16_t *gray = gray_all + img_idx * npixels;
        uint16_t *rgb = rgb_all + img_idx * npixels * 3;

        #pragma omp parallel for
        for (long y = 0; y < height; ++y) {
            for (long x = 0; x < width; ++x) {
                long pixel_idx = y * width + x;

                // Valore noto da CFA
                uint16_t c_u16 = gray[y * width + x];
                float c = static_cast<float>(c_u16);

                float Rf = 0.0f, Gf = 0.0f, Bf = 0.0f;

                // Schema RGGB:
                // (y%2==0, x%2==0) -> R
                // (y%2==0, x%2==1) -> G (riga R)
                // (y%2==1, x%2==0) -> G (riga B)
                // (y%2==1, x%2==1) -> B
                bool y_even = (y % 2 == 0);
                bool x_even = (x % 2 == 0);

                if (y_even && x_even) {
                    // ----- Pixel R -----
                    Rf = c;
                    Gf = apply_kernel_5x5_cpu(gray, width, height, x, y, KERNEL_G_at_RB);
                    Bf = apply_kernel_5x5_cpu(gray, width, height, x, y, KERNEL_RB_at_opposite);
                }
                else if (y_even && !x_even) {
                    // ----- Pixel G su riga R -----
                    Gf = c;
                    // R a green (red rows): kernel "diag"
                    Rf = apply_kernel_5x5_cpu(gray, width, height, x, y, KERNEL_RB_at_G_diag);
                    // B a green (red rows): kernel trasposto "cross"
                    Bf = apply_kernel_5x5_cpu(gray, width, height, x, y, KERNEL_RB_at_G_cross);
                }
                else if (!y_even && x_even) {
                    // ----- Pixel G su riga B -----
                    Gf = c;
                    // B a green (blue rows): kernel "diag"
                    Bf = apply_kernel_5x5_cpu(gray, width, height, x, y, KERNEL_RB_at_G_diag);
                    // R a green (blue rows): kernel trasposto "cross"
                    Rf = apply_kernel_5x5_cpu(gray, width, height, x, y, KERNEL_RB_at_G_cross);
                }
                else { // (!y_even && !x_even)
                    // ----- Pixel B -----
                    Bf = c;
                    Gf = apply_kernel_5x5_cpu(gray, width, height, x, y, KERNEL_G_at_RB);
                    Rf = apply_kernel_5x5_cpu(gray, width, height, x, y, KERNEL_RB_at_opposite);
                }

                // Clamp e scrittura planare
                rgb[pixel_idx]             = clamp_u16(Rf);
                rgb[pixel_idx + npixels]   = clamp_u16(Gf);
                rgb[pixel_idx + 2*npixels] = clamp_u16(Bf);
            }
        }
    }
}