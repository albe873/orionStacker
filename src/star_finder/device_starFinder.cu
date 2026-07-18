#ifndef CUDA_DEVICE_STARFINDER_CU
#define CUDA_DEVICE_STARFINDER_CU

#include <cuda_runtime.h>
#include "cuda_helper.h"

#include "star_finder.h"
#include "device_threshold.cu"
#include "device_otsu_centralized.cu"

__global__ void sum_brightness_planar_uint16(const u_int16_t *input_rgb, u_int64_t input_width, u_int64_t npixels, 
                                      star s, star_detail *sd) {
    // coordinates
    u_int64_t x = s.start_x + blockIdx.x * blockDim.x + threadIdx.x;
    u_int64_t y = s.start_y + blockIdx.y * blockDim.y + threadIdx.y;

    // boundary check
    if (x >= s.start_x + s.size_x || y >= s.start_y + s.size_y)
        return;

    // shared memory for sums
    __shared__ double shared_red_sum, shared_green_sum, shared_blue_sum;
    // initialize shared memory (only first thread per block)
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        shared_red_sum = 0;
        shared_green_sum = 0;
        shared_blue_sum = 0;
    }
    __syncthreads();    // synchronize threads of the block before shared memory use

    // add in shared memory
    u_int64_t idx = y * input_width + x;
    atomicAdd(&shared_red_sum,   (double) input_rgb[idx]);
    idx += npixels;
    atomicAdd(&shared_green_sum, (double) input_rgb[idx]);
    idx += npixels;
    atomicAdd(&shared_blue_sum,  (double) input_rgb[idx]);
    __syncthreads();

    // add to global memory
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(&sd->b_red,   shared_red_sum);
        atomicAdd(&sd->b_green, shared_green_sum);
        atomicAdd(&sd->b_blue,  shared_blue_sum);
        atomicAdd(&sd->b,       (0.299*shared_red_sum + 0.587*shared_green_sum + 0.114*shared_blue_sum));
    }
}

__global__ void baricenter_uint16(const u_int16_t *input_grayscale, u_int64_t input_width,
                                  star s, star_detail *sd) {
    // coordinates
    u_int64_t x = s.start_x + blockIdx.x * blockDim.x + threadIdx.x;
    u_int64_t y = s.start_y + blockIdx.y * blockDim.y + threadIdx.y;

    // boundary check
    if (x >= s.start_x + s.size_x || y >= s.start_y + s.size_y)
        return;

    u_int64_t idx = y * input_width + x;
    u_int16_t value = input_grayscale[idx];
    
    // Accumulate weighted positions
    double x_to_sum = (double) x * value / sd->b;
    double y_to_sum = (double) y * value / sd->b;

    // shared memory for sums
    __shared__ double shared_x_sum, shared_y_sum;
    // initialize shared memory (only first thread per block)
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        shared_x_sum = 0;
        shared_y_sum = 0;
    }
    __syncthreads();    // synchronize threads of the block before shared memory use

    // add in shared memory
    atomicAdd(&shared_x_sum, x_to_sum);
    atomicAdd(&shared_y_sum, y_to_sum);
    __syncthreads();

    // add to global memory
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(&sd->x, shared_x_sum);
        atomicAdd(&sd->y, shared_y_sum);
    }
}


// ---- Helper functions for star detection ----

__device__ inline int previous_dir(int dir) {
    return (dir == 0) ? 3 : dir - 1;
}

__device__ inline void change_direction(int8_t &dir, int8_t &dir_x_or_y) {
    // cambio direzione x o y (inverto sempre)
    dir_x_or_y = 1 - dir_x_or_y;

    // aumento dir, poi se ho completato un giro resetto i contatori
    dir++;
    if (dir == 4)
        dir = 0;
}

__device__ inline void write_star(star *stars, u_int32_t *num_stars, u_int32_t max_stars, 
                                  uint64_t start_x, uint64_t start_y,
                                  uint32_t size_x,  uint32_t size_y) {
    u_int32_t idx = atomicAdd(num_stars, 1);
    if (idx < max_stars) {
        stars[idx].start_x = start_x;
        stars[idx].start_y = start_y;
        stars[idx].size_x = size_x;
        stars[idx].size_y = size_y;
    }
    else {
        *num_stars = max_stars;
    }
}


// ---- Star detection kernel ---

__global__ void detect_stars(const u_int8_t *input, u_int64_t width, u_int64_t height,
                             u_int16_t max_star_size, u_int16_t min_star_size,
                             star *stars, u_int32_t *num_stars, u_int32_t max_stars) {
    // coordinates
    u_int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    u_int64_t y = blockIdx.y * blockDim.y + threadIdx.y;

    // boundary check
    if (x >= width || y >= height)
        return;

    // get the pixel index and the value
    u_int64_t idx = y * width + x;
    auto current = input[idx];
    
    // Se il pixel corrente è nero, non è una stella
    if (current == 0)
        return;

    bool is_star = true;
    bool all_black = true;                                  // una stella deve essere contenuta in un quadrato di pixel neri
    bool finished_dir[4] = {false, false, false, false};    // per ogni direzione, indica se ho finito di esplorare quella direzione

    const int8_t directions[4][2] = {{1,0},{0,1},{-1,0},{0,-1}};  // variazione x e y per ogni direzione

    int32_t   stepCount = 0;                              // contatore dei passi fatti nella direzione corrente
    int32_t   stepCurrentLimit[2] = {1, 1};               // step limit per x e y
    int8_t    dir = 0;                                    // direzione corrente
    int8_t    dir_x_or_y = 0;                             // = 0 se x, = 1 se y, per comodità

    u_int64_t current_idx;                                // indice del pixel corrente

    // salvo le coordinate iniziali
    u_int64_t min_x = x;
    u_int64_t min_y = y;

    while(stepCurrentLimit[0] < max_star_size && stepCurrentLimit[1] < max_star_size) {

        // Controllo se ho completato un lato
        if(stepCount == stepCurrentLimit[dir_x_or_y]) {
            stepCount = 0;

            // Incremento il limite di passi della direzione corrente
            // solo se non ho finito la direzione precedente
            // es. se ho finito il lato in basso, non devo incrementare il limite 
            if (!finished_dir[previous_dir(dir)])
                stepCurrentLimit[dir_x_or_y]++;

            // Controllo se tutti i pixel sono neri, allora finisco la ricerca della stella nella direzione
            if (all_black && stepCurrentLimit[dir_x_or_y] > min_star_size) {
                finished_dir[dir] = true;

                // controllo se ho finito tutte le direzioni
                if (finished_dir[0] && finished_dir[1] && finished_dir[2] && finished_dir[3])
                    break;
            }

            change_direction(dir, dir_x_or_y);

            all_black = true;
        }

        // se i pixel del lato corrente sono neri, allora non esploro più in quella direzione e passo alla sucessiva
        if (finished_dir[dir]) {
            // imposto di quanto mi devo muovere 
            stepCount = stepCurrentLimit[dir_x_or_y];

            // mi muovo, saltando tutti i controlli e vado al ciclo sucessivo
            x += directions[dir][0] * stepCount;
            y += directions[dir][1] * stepCount;
            if (x >= width-1 || y >= height-1) {
                is_star = false;
                break;
            }

            continue;
        }

        // mi muovo di un passo
        x += directions[dir][0];
        y += directions[dir][1];
        stepCount++;

        // check se sono ai bordi dell'immagine
        if (x >= width || y >= height) {
            is_star = false;
            break;
        }     
        
        // aggiorno le coordinate minime
        min_x = min(x, min_x);
        min_y = min(y, min_y);

        // Controlla se il pixel corrente è maggiore del pixel centrale
        // se è maggiore allora non è il centro di una stella, esco dal ciclo
        current_idx = y * width + x;
        if (input[current_idx] > current) {
            is_star = false;
            break;
        }
        
        // Controllo se non c'è un pixel candidato come centro con stessa luminosità
        // se esiste, allora controllo idx, se idx è maggiore di inizial_idx allora non è il centro di una stella, esco dal ciclo 
        if (input[current_idx] == current) {
            if (current_idx > idx) {
                is_star = false;
                break;
            }
        }

        // Controllo se il pixel corrente è nero, se non lo è imposto all_black a false
        if (all_black && input[current_idx] > 0)
            all_black = false;
    }

    // Verifica le condizioni per essere una stella
    // 1 variabile is_star = true
    // 2 tutti i lati devono essere finiti
    // 3 dimensione massima non deve essere raggiunta
    if (is_star && 
        finished_dir[0] && finished_dir[1] && finished_dir[2] && finished_dir[3] &&
        stepCurrentLimit[0] < max_star_size && stepCurrentLimit[1] < max_star_size)
    {
        //printf("Star detected at (%lu, %lu) with size %u, idx= %lu\n", min_x, height - min_y, final_dim, idx);
        //baricenter_uint16<<<1, 256>>>(input, &x_center, &y_center, &sum, min_x, min_y, stepCurrentLimit[0], stepCurrentLimit[1]);
        write_star(stars, num_stars, max_stars, min_x, min_y, stepCurrentLimit[0], stepCurrentLimit[1]);     
    }
}


// ---- Host function to launch the kernels ----

void to_grayscale_planar_gpu(const u_int16_t *img, u_int16_t *img_gray, u_int64_t npixels) {
    dim3 block_size_1d(256);
    dim3 grid_size_1d((npixels + block_size_1d.x - 1) / block_size_1d.x);

    to_grayscale_planar_uint16<<<grid_size_1d, block_size_1d>>>(img, img_gray, npixels, npixels * 2);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}

void compute_threshold_gpu( const u_int16_t *img, u_int8_t *out_img,
                            u_int64_t width, u_int64_t height, 
                            threshold_params params) {
    u_int64_t npixels = width * height;
    // in case of fast adaptive thresholding, allocate reduced image
    u_int16_t *reduced_image = nullptr;
    if (params.type == TR_FAST_ADAPTIVE) {
        CHECK(cudaMallocManaged(&reduced_image, (npixels / params.reduce_factor / params.reduce_factor) * sizeof(u_int16_t)));
        //CHECK(cudaMemPrefetchAsync(reduced_image, (npixels / reduce_factor / reduce_factor) * sizeof(u_int16_t), devLoc, 0));
    }

    dim3 block_size_1d(256);
    dim3 grid_size_1d((npixels / 2 + block_size_1d.x - 1) / block_size_1d.x);

    dim3 block_size_2d(16, 16);
    dim3 grid_size_2d(  (width + block_size_2d.x - 1) / block_size_2d.x, 
                        (height + block_size_2d.y - 1) / block_size_2d.y
                    );

    // --- Apply thresholding ---
    switch (params.type) {
        case TR_SIMPLE:
            simple_threshold_uint16<<<grid_size_1d, block_size_1d>>>(img, out_img, npixels, params.threshold);
            CHECK(cudaGetLastError());
            break;
        case TR_ADAPTIVE:
            adaptive_threshold<uint16_t><<<grid_size_2d, block_size_2d>>>(img, out_img, width, height, params.window_size / 2, params.threshold);
            CHECK(cudaGetLastError());
            break;
        case TR_FAST_ADAPTIVE:
            reduce_image<uint16_t><<<grid_size_2d, block_size_2d>>>(img, reduced_image, width, height,
                                                                    width / params.reduce_factor, height / params.reduce_factor, // new dimensions
                                                                    params.reduce_factor, params.reduce_factor * params.reduce_factor); // reduce factor and squared reduce factor
            CHECK(cudaGetLastError());
            CHECK(cudaDeviceSynchronize());
            adaptive_threshold_approximate<uint16_t><<<grid_size_2d, block_size_2d>>>(
                img, out_img, width, height, reduced_image, width /  params.reduce_factor,
                params.reduce_factor, params.window_size, params.threshold);
            CHECK(cudaGetLastError());
            break;
        case OTSU_CENTRALIZED:
            cuda_otsu_centralized_threshold(img, out_img, width, height, params.window_size, params.threshold_scale);
            
    }
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    
    // free allocated memory
    if (params.type == TR_FAST_ADAPTIVE) {
        CHECK(cudaFree(reduced_image));
    }
}


void populate_star_details_gpu(star_detail *stars_details, star *stars, u_int32_t n_stars, 
                               const u_int16_t *img_rgb, const u_int16_t *img_gray,
                               u_int64_t width, u_int64_t npixels) {
    // for each star, first I compute brightness then the baricenter
    // (I need the brightness sums to compute the baricenter)
    for (u_int32_t i = 0; i < n_stars; i++) {
        init_star_detail(&stars_details[i]);
        
        // dimensions
        dim3 block_size_star(16, 16);
        dim3 grid_size_star(  (stars[i].size_x + block_size_star.x - 1) / block_size_star.x, 
                              (stars[i].size_y + block_size_star.y - 1) / block_size_star.y
                            );
        // kernel launch
        sum_brightness_planar_uint16<<<grid_size_star, block_size_star>>>(img_rgb, width, npixels, stars[i], &stars_details[i]);
    } 
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    for (u_int32_t i = 0; i < n_stars; i++) {
        dim3 block_size_star(16, 16);
        dim3 grid_size_star(  (stars[i].size_x + block_size_star.x - 1) / block_size_star.x, 
                              (stars[i].size_y + block_size_star.y - 1) / block_size_star.y
                            );
        baricenter_uint16<<<grid_size_star, block_size_star>>>(img_gray, width, stars[i], &stars_details[i]);
    }
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}


void detect_stars_gpu(const u_int8_t *img, u_int64_t width, u_int64_t height,
                      u_int16_t max_star_size, u_int16_t min_star_size,
                      star *stars, u_int32_t *num_stars, u_int32_t max_stars) {
    dim3 block_size_2d(16, 16);
    dim3 grid_size_2d(  (width + block_size_2d.x - 1) / block_size_2d.x, 
                        (height + block_size_2d.y - 1) / block_size_2d.y
                    );
    detect_stars<<<grid_size_2d, block_size_2d>>>(img, width, height, max_star_size, min_star_size, stars, num_stars, max_stars);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}

#endif // CUDA_DEVICE_STARFINDER_CU