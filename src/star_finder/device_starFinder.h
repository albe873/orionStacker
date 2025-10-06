#ifndef CUDA_DEVICE_THRESHOLDING_H
#define CUDA_DEVICE_THRESHOLDING_H

#define MIN_STAR_SIZE 6

// fits data is in planar format
// calculating two pixels at a time to improve cache hit rate
__global__ void to_grayscale_planar_uint16(const u_int16_t *image, u_int16_t *gray_image, const u_int64_t npixels) {
    const u_int64_t idx1 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const u_int64_t idx2 = idx1 + 1;
    const u_int64_t npixels2 = 2 * npixels;

    if (idx2 < npixels) {
        u_int16_t red1 = image[idx1];
        u_int16_t red2 = image[idx2];

        u_int16_t green1 = image[idx1 + npixels];
        u_int16_t green2 = image[idx2 + npixels];

        u_int16_t blue1 = image[idx1 + npixels2];
        u_int16_t blue2 = image[idx2 + npixels2];

        gray_image[idx1] = 0.299*red1 + 0.587*green1 + 0.114*blue1;
        gray_image[idx2] = 0.299*red2 + 0.587*green2 + 0.114*blue2;
    }
    else if (idx2 == npixels) {
        u_int8_t red1 = image[idx1];
        u_int8_t green1 = image[idx1 + npixels];
        u_int8_t blue1 = image[idx1 + npixels2];
        gray_image[idx1] = 0.299*red1 + 0.587*green1 + 0.114*blue1;
    }
}

__global__ void to_grayscale_planar_uint8(const u_int8_t *image, u_int8_t *gray_image, const u_int64_t npixels) {
    const u_int64_t idx1 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const u_int64_t idx2 = idx1 + 1;
    const u_int64_t idx3 = idx1 + 2;
    const u_int64_t idx4 = idx1 + 3;
    const u_int64_t npixels2 = 2 * npixels;

    if (idx2 < npixels) {
        u_int8_t red1 = image[idx1];
        u_int8_t red2 = image[idx2];
        u_int8_t red3 = image[idx3];
        u_int8_t red4 = image[idx4];

        u_int8_t green1 = image[idx1 + npixels];
        u_int8_t green2 = image[idx2 + npixels];
        u_int8_t green3 = image[idx3 + npixels];
        u_int8_t green4 = image[idx4 + npixels];

        u_int8_t blue1 = image[idx1 + npixels2];
        u_int8_t blue2 = image[idx2 + npixels2];
        u_int8_t blue3 = image[idx3 + npixels2];
        u_int8_t blue4 = image[idx4 + npixels2];

        gray_image[idx1] = 0.299*red1 + 0.587*green1 + 0.114*blue1;
        gray_image[idx2] = 0.299*red2 + 0.587*green2 + 0.114*blue2;
        gray_image[idx3] = 0.299*red3 + 0.587*green3 + 0.114*blue3;
        gray_image[idx4] = 0.299*red4 + 0.587*green4 + 0.114*blue4;
    }
    else if (idx2 == npixels) {
        u_int8_t red1 = image[idx1];
        u_int8_t green1 = image[idx1 + npixels];
        u_int8_t blue1 = image[idx1 + 2*npixels];
        gray_image[idx1] = 0.299*red1 + 0.587*green1 + 0.114*blue1;
    }
    else if (idx3 == npixels) {
        u_int8_t red1 = image[idx1];
        u_int8_t red2 = image[idx2];
        u_int8_t green1 = image[idx1 + npixels];
        u_int8_t green2 = image[idx2 + npixels];
        u_int8_t blue1 = image[idx1 + npixels2];
        u_int8_t blue2 = image[idx2 + npixels2];
        gray_image[idx1] = 0.299*red1 + 0.587*green1 + 0.114*blue1;
        gray_image[idx2] = 0.299*red2 + 0.587*green2 + 0.114*blue2;
    }
    else if (idx4 == npixels) {
        u_int8_t red1 = image[idx1];
        u_int8_t red2 = image[idx2];
        u_int8_t red3 = image[idx3];
        u_int8_t green1 = image[idx1 + npixels];
        u_int8_t green2 = image[idx2 + npixels];
        u_int8_t green3 = image[idx3 + npixels];
        u_int8_t blue1 = image[idx1 + npixels2];
        u_int8_t blue2 = image[idx2 + npixels2];
        u_int8_t blue3 = image[idx3 + npixels2];
        gray_image[idx1] = 0.299*red1 + 0.587*green1 + 0.114*blue1;
        gray_image[idx2] = 0.299*red2 + 0.587*green2 + 0.114*blue2;
        gray_image[idx3] = 0.299*red3 + 0.587*green3 + 0.114*blue3;
    }
}

// calculating two pixels at a time to improve cache hit rate
__global__ void simple_threshold_uint16(const u_int16_t *image, u_int16_t *output, const u_int64_t npixels, const u_int16_t threshold) {
    u_int64_t idx1 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    u_int64_t idx2 = idx1 + 1;

    if (idx2 < npixels) {
        u_int16_t val1 = image[idx1];
        u_int16_t val2 = image[idx2];

        output[idx1] = val1 > threshold ? val1 : 0;
        output[idx2] = val2 > threshold ? val2 : 0;
    }
    else if (idx2 == npixels) {
        u_int16_t val1 = image[idx1];
        output[idx1] = val1 > threshold ? val1 : 0;
    }
}

__global__ void simple_threshold_uint8(const u_int8_t *image, u_int8_t *output, const u_int64_t npixels, const u_int8_t threshold) {
    const u_int64_t idx1 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const u_int64_t idx2 = idx1 + 1;
    const u_int64_t idx3 = idx1 + 2;
    const u_int64_t idx4 = idx1 + 3;

    if (idx2 < npixels) {
        u_int8_t val1 = image[idx1];
        u_int8_t val2 = image[idx2];
        u_int8_t val3 = image[idx3];
        u_int8_t val4 = image[idx4];

        output[idx1] = val1 > threshold ? val1 : 0;
        output[idx2] = val2 > threshold ? val2 : 0;
        output[idx3] = val3 > threshold ? val3 : 0;
        output[idx4] = val4 > threshold ? val4 : 0;
    }
    else if (idx2 == npixels) {
        u_int8_t val1 = image[idx1];
        output[idx1] = val1 > threshold ? val1 : 0;
    }
    else if (idx3 == npixels) {
        u_int8_t val1 = image[idx1];
        u_int8_t val2 = image[idx2];
        output[idx1] = val1 > threshold ? val1 : 0;
        output[idx2] = val2 > threshold ? val2 : 0;
    }
    else if (idx4 == npixels) {
        u_int8_t val1 = image[idx1];
        u_int8_t val2 = image[idx2];
        u_int8_t val3 = image[idx3];
        output[idx1] = val1 > threshold ? val1 : 0;
        output[idx2] = val2 > threshold ? val2 : 0;
        output[idx3] = val3 > threshold ? val3 : 0;
    }
}

#define DEFINE_ADAPTIVE_THRESHOLD_KERNEL(FUNC_NAME, TYPE) \
__global__ void FUNC_NAME(const TYPE *image, TYPE *output, const u_int64_t width, const u_int64_t height, \
                          u_int16_t windowSize, const TYPE offset) { \
    const u_int64_t x = blockIdx.x * blockDim.x + threadIdx.x; \
    const u_int64_t y = blockIdx.y * blockDim.y + threadIdx.y; \
    \
    if (x < width && y < height) { \
        windowSize /= 2; \
        const u_int64_t startX = max(x - windowSize, 0L); \
        const u_int64_t endX = min(x + windowSize, width); \
        const u_int64_t startY = max(y - windowSize, 0L); \
        const u_int64_t endY = min(y + windowSize, height); \
        \
        u_int64_t sum = 0; \
        for (u_int64_t i = startY; i < endY; i++) { \
            for (u_int64_t j = startX; j < endX; j++) { \
                sum += image[i * width + j]; \
            } \
        } \
        u_int64_t localMean = sum / ((endX - startX) * (endY - startY)); \
        TYPE pixel = image[y * width + x]; \
        \
        output[y * width + x] = (pixel > (localMean + offset)) ? pixel : 0; \
    } \
}

// uint8_t version of the kernel
DEFINE_ADAPTIVE_THRESHOLD_KERNEL(adaptive_threshold_uint8, u_int8_t)
// uint16_t version of the kernel
DEFINE_ADAPTIVE_THRESHOLD_KERNEL(adaptive_threshold_uint16, u_int16_t)


// Ogni thread si occupa di un pixel dell'immagine ridotta.
#define DEFINE_REDUCE_IMAGE_KERNEL(FUNC_NAME, TYPE) \
__global__ void FUNC_NAME(TYPE *image, TYPE *reduced_image, u_int64_t width, u_int64_t height, u_int16_t reduce_factor) { \
    u_int64_t x = blockIdx.x * blockDim.x + threadIdx.x; \
    u_int64_t y = blockIdx.y * blockDim.y + threadIdx.y; \
    \
    u_int64_t new_width = width / reduce_factor; \
    u_int64_t new_height = height / reduce_factor; \
    \
    if (x < new_width && y < new_height) { \
        u_int64_t sum = 0; \
        u_int32_t out_of_range = 0; \
        for (u_int16_t i = 0; i < reduce_factor; i++) { \
            for (u_int16_t j = 0; j < reduce_factor; j++) { \
                u_int64_t orig_x = x * reduce_factor + i; \
                u_int64_t orig_y = y * reduce_factor + j; \
                if (orig_x >= width || orig_y >= height) { \
                    out_of_range++; \
                    continue; \
                } \
                sum += image[orig_y * width + orig_x]; \
            } \
        } \
        reduced_image[y * new_width + x] = sum / (reduce_factor * reduce_factor - out_of_range); \
    } \
}

// uint16_t version of the kernel
DEFINE_REDUCE_IMAGE_KERNEL(reduce_image_uint16, u_int16_t)

// uint8_t version of the kernel
DEFINE_REDUCE_IMAGE_KERNEL(reduce_image_uint8, u_int8_t)



#define DEFINE_ADAPTIVE_THRESHOLD_APPROXIMATE_KERNEL(FUNC_NAME, TYPE) \
__global__ void FUNC_NAME( \
            TYPE *image, TYPE *output, u_int64_t width, u_int64_t height, \
            TYPE *reduced_image, u_int16_t reduce_factor, u_int16_t windowSize, TYPE offset \
    ) \
{ \
    u_int64_t x = blockIdx.x * blockDim.x + threadIdx.x; \
    u_int64_t y = blockIdx.y * blockDim.y + threadIdx.y; \
    \
    if (x < width && y < height) { \
        u_int16_t halfWindow = windowSize / 2; \
        u_int64_t startX = (x > halfWindow) ? (x - halfWindow) / reduce_factor : 0; \
        u_int64_t endX = min((u_int64_t)(x + halfWindow), width - 1) / reduce_factor; \
        u_int64_t startY = (y > halfWindow) ? (y - halfWindow) / reduce_factor : 0; \
        u_int64_t endY = min((u_int64_t)(y + halfWindow), height - 1) / reduce_factor; \
        \
        u_int64_t sum = 0; \
        u_int64_t reduced_width = width / reduce_factor; \
        for (u_int64_t i = startY; i <= endY; i++) { \
            for (u_int64_t j = startX; j <= endX; j++) { \
                sum += reduced_image[i * reduced_width + j]; \
            } \
        } \
        \
        u_int64_t num_pixels = (endX - startX + 1) * (endY - startY + 1); \
        TYPE localMean = (num_pixels > 0) ? (sum / num_pixels) : 0; \
        TYPE pixel = image[y * width + x]; \
        \
        output[y * width + x] = (pixel > (localMean + offset)) ? pixel : 0; \
    } \
}

// uint16_t version of the kernel
DEFINE_ADAPTIVE_THRESHOLD_APPROXIMATE_KERNEL(adaptive_threshold_approximate_uint16, u_int16_t)

// uint8_t version of the kernel
DEFINE_ADAPTIVE_THRESHOLD_APPROXIMATE_KERNEL(adaptive_threshold_approximate_uint8, u_int8_t)


__device__ inline int previous_dir(int dir) {
    return (dir == 0) ? 3 : dir - 1;
}

__device__ inline void draw_rectangle(u_int16_t *output, u_int64_t width, u_int64_t min_x, u_int64_t min_y, u_int32_t dim_x, u_int32_t dim_y) {
        // Disegna il quadrato a partire dalle coordinate minime
        
        // ciclo sui lati orizzontali
        u_int64_t idx1 = min_y * width + min_x;
        u_int64_t idx2 = (min_y + dim_y) * width + min_x;
        for (int i = 0; i < dim_x; i++) {
            output[idx1] = 65535;
            output[idx2] = 65535;
            // basta un incremento unitario
            idx1++;
            idx2++;
        }

        // ciclo sui lati verticali
        idx1 = min_y * width + min_x;
        idx2 = min_y * width + min_x + dim_x;
        for (int j = 0; j < dim_y; j++) {
            output[idx1] = 65535;
            output[idx2] = 65535;
            // devo cambiare riga, incremento pari alla larghezza della riga
            idx1 += width;
            idx2 += width;
        }
        output[idx2] = 65535; // segna l'angolo in alto a destra
}



// Kernel per la rilevazione delle stelle
__global__ void detect_stars_uint16(u_int16_t *input, u_int16_t *output, u_int64_t width, u_int64_t height, u_int16_t max_star_size, u_int16_t min_star_size) {
    // coordinates
    u_int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    u_int64_t y = blockIdx.y * blockDim.y + threadIdx.y;

    // boundary check
    if (x >= width || y >= height)
        return;

    // get the pixel index and the value
    u_int64_t idx = y * width + x;
    u_int16_t current = input[idx];
    
    // Se il pixel corrente è nero, non è una stella
    if (current == 0)
        return;

    bool is_star = true;
    bool all_black = true;                                  // una stella deve essere contenuta in un quadrato di pixel neri
    bool finished_dir[4] = {false, false, false, false};    // per ogni direzione, indica se ho finito di esplorare quella direzione

    int8_t directions[4][2] = {{1,0},{0,1},{-1,0},{0,-1}};  // variazione x e y per ogni direzione

    int32_t   stepCount = 0;                              // contatore dei passi fatti nella direzione corrente
    int32_t   stepCurrentLimit[2] = {1, 1};               // step limit per x e y
    int8_t    dir = 0;                                    // direzione corrente
    int8_t    dir_x_or_y = 0;                             // = 0 se x, = 1 se y, per comodità

    u_int64_t current_idx;                                // indice del pixel corrente

    // salvo le coordinate iniziali
    u_int64_t start_x = x;
    u_int64_t start_y = y;

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


            // cambio direzione x o y (inverto sempre)
            dir_x_or_y = 1 - dir_x_or_y;
            dir++;
            
            // Controllo se ho completato un giro, allora resetto i contatori
            if (dir == 4)
                dir = 0;
                        
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

        // check se sono ai bordi dell'immagine
        if (x >= width || y >= height) {
            is_star = false;
            break;
        }     
        // incremento il contatore dei passi fatti
        stepCount++;
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

    // Usa il MINIMO dei limiti finali per definire la dimensione
    u_int32_t final_dim = min(stepCurrentLimit[0], stepCurrentLimit[1]);

    // Verifica le condizioni per essere una stella
    // 1 variabile is_star = true
    // 2 tutti i lati devono essere finiti
    // 3 dimensione massima non deve essere raggiunta
    if (is_star && 
        finished_dir[0] && finished_dir[1] && finished_dir[2] && finished_dir[3] &&
        stepCurrentLimit[0] < max_star_size && stepCurrentLimit[1] < max_star_size)
    {
        printf("Star detected at (%lu, %lu) with size %u, idx= %lu\n", start_x, height - start_y, final_dim, idx);
        draw_rectangle(output, width, min_x, min_y, stepCurrentLimit[0], stepCurrentLimit[1]);
    }
}

#endif // CUDA_DEVICE_THRESHOLDING_H