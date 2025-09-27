#ifndef MHC_APPLY_H
#define MHC_APPLY_H

#include "MHC_filters.h"

// Funzione inline per clamp
__device__ inline int clamp_index(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// ============================================================================
// Applica un kernel 5x5 centrato in (x,y) alla CFA (grayscale).
// - gray: puntatore CFA
// - width, height: dimensioni immagine
// - x, y: coordinate pixel corrente
// - kernel: matrice 5x5 (float) con i coefficienti
// Ritorna il valore filtrato (float).
// ============================================================================
__device__ inline float apply_kernel_5x5(
    const u_int16_t *gray,
    int width,
    int height,
    int x,
    int y,
    const float kernel[5][5]
) {
    float acc = 0.0f;

    #pragma unroll 5
    for (int ky = -2; ky <= 2; ky++) {
        int yy = clamp_index(y + ky, 0, height - 1);

        #pragma unroll 5
        for (int kx = -2; kx <= 2; kx++) {
            int xx = clamp_index(x + kx, 0, width - 1);

            float coeff = kernel[ky + 2][kx + 2];   // indice da 0..4
            float val   = (float) gray[yy * width + xx];

            acc += coeff * val;
        }
    }

    // Normalizzazione: tutti i kernel Malvar sono divisi per 8
    return acc / 8.0f;
}

#endif
