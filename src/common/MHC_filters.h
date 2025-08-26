#ifndef MHC_FILTERS_H
#define MHC_FILTERS_H

// ============================================================================
// Malvar-He-Cutler demosaicing kernels (5x5) - RGGB Bayer pattern
// Coefficient values are scaled by 1/8 (come nel paper).
// Ogni kernel si applica a seconda del colore del pixel centrale.
// ============================================================================

// ---------------------------------------------------------------------------
// 1. Verde ai pixel R o B
// Usato quando il pixel centrale è R o B, per stimare G mancante
__device__ __constant__ float KERNEL_G_at_RB[5][5] = {
    {  0.0f,  0.0f, -1.0f,  0.0f,  0.0f },
    {  0.0f,  0.0f,  2.0f,  0.0f,  0.0f },
    { -1.0f,  2.0f,  4.0f,  2.0f, -1.0f },
    {  0.0f,  0.0f,  2.0f,  0.0f,  0.0f },
    {  0.0f,  0.0f, -1.0f,  0.0f,  0.0f }
};
// divisore = 8

// ---------------------------------------------------------------------------
// 2. Rosso ai pixel G (riga R) 
// 4. Blu ai pixel G (riga B)
// NOTA: stesso kernel usato, cambia il canale da ricostruire.
__device__ __constant__ float KERNEL_RB_at_G_diag[5][5] = {
    {  0.0f,  0.0f,  0.5f,  0.0f,  0.0f },
    {  0.0f, -1.0f,  0.0f, -1.0f,  0.0f },
    { -1.0f,  4.0f,  5.0f,  4.0f, -1.0f },
    {  0.0f, -1.0f,  0.0f, -1.0f,  0.0f },
    {  0.0f,  0.0f,  0.5f,  0.0f,  0.0f }
};
// divisore = 8

// ---------------------------------------------------------------------------
// 3. Rosso ai pixel G (colonna R) 
// 5. Blu ai pixel G (colonna B)
// È il trasposto del precedente.
__device__ __constant__ float KERNEL_RB_at_G_cross[5][5] = {
    {  0.0f,  0.0f, -1.0f,  0.0f,  0.0f },
    {  0.0f, -1.0f,  4.0f, -1.0f,  0.0f },
    {  0.5f,  0.0f,  5.0f,  0.0f,  0.5f },
    {  0.0f, -1.0f,  4.0f, -1.0f,  0.0f },
    {  0.0f,  0.0f, -1.0f,  0.0f,  0.0f }
};
// divisore = 8
// Nota: questo è semplicemente la trasposizione di KERNEL_RB_at_G_diag

// ---------------------------------------------------------------------------
// 6. Rosso ai pixel B 
// 8. Blu ai pixel R
__device__ __constant__ float KERNEL_RB_at_opposite[5][5] = {
    {  0.0f,  0.0f, -1.5f,  0.0f,  0.0f },
    {  0.0f,  2.0f,  0.0f,  2.0f,  0.0f },
    { -1.5f,  0.0f,  6.0f,  0.0f, -1.5f },
    {  0.0f,  2.0f,  0.0f,  2.0f,  0.0f },
    {  0.0f,  0.0f, -1.5f,  0.0f,  0.0f }
};
// divisore = 8

#endif
