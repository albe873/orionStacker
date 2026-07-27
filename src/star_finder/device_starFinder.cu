#include "cuda_helper.hh"

#include "star_finder.hh"
#include "device_threshold.cu"
#include "device_otsu_centralized.cu"
#include "device_warp.cu"

#define BLOCK_SIZE_1D   256
#define BLOCK_DIM_2D    16
#define BLOCK_SIZE_2D   BLOCK_DIM_2D*BLOCK_DIM_2D
#define WARP_SIZE       32

// Shared memory reduction for double (tree-based, no atomics)
__inline__ __device__ double warp_reduce_sum(double val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__ double block_reduce_sum(double val) {
    __shared__ double shared[BLOCK_SIZE_1D / WARP_SIZE]; // one per warp
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);
    if (lane == 0)
        shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0;
    if (wid == 0)
        val = warp_reduce_sum(val);
    return val;
}

// Fused kernel: brightness AND baricenter in one pass
__global__ void kernel_compute_star_details_planar(
    const uint16_t* __restrict__ input_rgb,
    const uint16_t* __restrict__ input_grayscale,
    uint64_t width,
    uint64_t npixels,
    const star* __restrict__ stars,
    star_detail* __restrict__ star_details,
    uint32_t n_stars)
{   
    // Each block handles one star
    star s = stars[blockIdx.x];
    star_detail local = {0, 0, 0, 0, 0, 0};

    uint64_t pixels_per_star = (uint64_t)s.size_x * s.size_y;
    // loop to handle stars with more pixels than BLOCK_SIZE_1D
    for (uint64_t tid = threadIdx.x; tid < pixels_per_star; tid += BLOCK_SIZE_1D) {
        uint64_t px = tid % s.size_x;   // relative coordinates
        uint64_t py = tid / s.size_x;
        uint64_t x = s.start_x + px;    // global coordinates
        uint64_t y = s.start_y + py;
        uint64_t idx = y * width + x;   // idx

        auto gray_val = input_grayscale[idx];

        // RGB channels
        auto r = input_rgb[idx];
        auto g = input_rgb[idx + npixels];
        auto b = input_rgb[idx + 2 * npixels];
        float brightness = 0.299f*r + 0.587f*g + 0.114f*b;

        local.b_red   += r;
        local.b_green += g;
        local.b_blue  += b;
        local.b       += brightness;
        local.x       += (double)x * gray_val;
        local.y       += (double)y * gray_val;
    }

    // Tree reduction within the block
    local.b_red   = block_reduce_sum(local.b_red);
    local.b_green = block_reduce_sum(local.b_green);
    local.b_blue  = block_reduce_sum(local.b_blue);
    local.b       = block_reduce_sum(local.b);
    local.x       = block_reduce_sum(local.x);
    local.y       = block_reduce_sum(local.y);

    // write to global memory
    if (threadIdx.x == 0) {
        double inv_b = 1.0 / local.b;
        star_details[blockIdx.x].x       = local.x * inv_b;
        star_details[blockIdx.x].y       = local.y * inv_b;
        star_details[blockIdx.x].b_red   = local.b_red;
        star_details[blockIdx.x].b_green = local.b_green;
        star_details[blockIdx.x].b_blue  = local.b_blue;
        star_details[blockIdx.x].b       = local.b;
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

__device__ inline void write_star(star *stars, uint32_t *num_stars, uint32_t max_stars, 
                                  uint64_t start_x, uint64_t start_y,
                                  uint32_t size_x,  uint32_t size_y) {
    uint32_t idx = atomicAdd(num_stars, 1);
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

__global__ void kernel_detect_stars(const uint8_t *input, uint64_t width, uint64_t height,
                                    uint16_t max_star_size, uint16_t min_star_size,
                                    star *stars, uint32_t *num_stars, uint32_t max_stars) {
    // coordinates
    uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t y = blockIdx.y * blockDim.y + threadIdx.y;

    // boundary check
    if (x >= width || y >= height)
        return;

    // get the pixel index and the value
    uint64_t idx = y * width + x;
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

    uint64_t current_idx;                                 // indice del pixel corrente

    // salvo le coordinate iniziali
    uint64_t min_x = x;
    uint64_t min_y = y;

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

void to_grayscale_planar_gpu(const uint16_t *img, uint16_t *img_gray, uint64_t npixels) {
    dim3 block_size_1d(BLOCK_SIZE_1D);
    dim3 grid_size_1d((npixels + block_size_1d.x - 1) / block_size_1d.x);

    kernel_grayscale_planar<<<grid_size_1d, block_size_1d>>>(img, img_gray, npixels, npixels * 2);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}

void compute_threshold_gpu(const uint16_t *img, uint8_t *out_img,
                           uint64_t width, uint64_t height, 
                           threshold_params params) {
    uint64_t npixels = width * height;
    // in case of fast adaptive thresholding, allocate reduced image
    uint16_t *reduced_image = nullptr;
    if (params.type == TR_FAST_ADAPTIVE) {
        CHECK(cudaMallocManaged(&reduced_image, (npixels / params.reduce_factor / params.reduce_factor) * sizeof(uint16_t)));
        //CHECK(cudaMemPrefetchAsync(reduced_image, (npixels / reduce_factor / reduce_factor) * sizeof(u_int16_t), devLoc, 0));
    }

    dim3 b_1d(BLOCK_SIZE_1D);
    dim3 g_1d((npixels / 2 + b_1d.x - 1) / b_1d.x);

    dim3 b_2d(BLOCK_DIM_2D, BLOCK_DIM_2D);
    dim3 g_2d((width  + b_2d.x - 1) / b_2d.x, 
              (height + b_2d.y - 1) / b_2d.y);

    // --- Apply thresholding ---
    switch (params.type) {
        case TR_SIMPLE:
            kernel_simple_threshold<<<g_1d, b_1d>>>(img, out_img, npixels, params.threshold);
            CHECK(cudaGetLastError());
            break;
        case TR_ADAPTIVE:
            kernel_adaptive_threshold<uint16_t><<<g_2d, b_2d>>>(
                img, out_img, width, height, params.window_size / 2, params.threshold);
            CHECK(cudaGetLastError());
            break;
        case TR_FAST_ADAPTIVE:
            kernel_reduce_image<uint16_t><<<g_2d, b_2d>>>(
                img, reduced_image, width, height,
                width / params.reduce_factor, height / params.reduce_factor, // new dimensions
                params.reduce_factor, params.reduce_factor * params.reduce_factor); // reduce factor and squared reduce factor
            CHECK(cudaGetLastError());
            CHECK(cudaDeviceSynchronize());
            kernel_adaptive_threshold_approximate<uint16_t><<<g_2d, b_2d>>>(
                img, out_img, width, height, reduced_image, width /  params.reduce_factor,
                params.reduce_factor, params.window_size, params.threshold);
            CHECK(cudaGetLastError());
            break;
        case OTSU_CENTRALIZED:
            otsu_centralized_threshold_gpu(img, out_img, width, height, params.window_size, params.threshold_scale);
            
    }
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    
    // free allocated memory
    if (params.type == TR_FAST_ADAPTIVE) {
        CHECK(cudaFree(reduced_image));
    }
}


void populate_star_details_gpu(star_detail *stars_details, star *stars, uint32_t n_stars, 
                               const uint16_t *img_rgb, const uint16_t *img_gray,
                               uint64_t width, uint64_t npixels) {
    for (uint32_t i = 0; i < n_stars; i++)
        init_star_detail(&stars_details[i]);

    kernel_compute_star_details_planar<<<n_stars, BLOCK_SIZE_1D>>>(
        img_rgb, img_gray, width, npixels, stars, stars_details, n_stars);
    CHECK(cudaDeviceSynchronize());
}


void detect_stars_gpu(const uint8_t *img, uint64_t width, uint64_t height,
                      uint16_t max_star_size, uint16_t min_star_size,
                      star *stars, uint32_t *num_stars, uint32_t max_stars) {
    // Initialize num_stars to 0 before detection
    CHECK(cudaMemset(num_stars, 0, sizeof(uint32_t)));
    dim3 b_2d(16, 16);
    dim3 g_2d((width  + b_2d.x - 1) / b_2d.x, 
              (height + b_2d.y - 1) / b_2d.y);
    kernel_detect_stars<<<g_2d, b_2d>>>(img, width, height, max_star_size, min_star_size, stars, num_stars, max_stars);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}
