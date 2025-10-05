#ifndef CUDA_DEVICE_THRESHOLDING_H
#define CUDA_DEVICE_THRESHOLDING_H

#define MIN_STAR_SIZE 6

// fits data is in planar format
// calculating two pixels at a time to improve cache hit rate
__global__ void to_grayscale_fits(u_int16_t *image, u_int16_t *gray_image, u_int64_t npixels) {
    u_int64_t idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    idx1 *= 2;
    u_int64_t idx2 = idx1 + 1;

    if (idx2 < npixels) {
        u_int16_t red1 = image[idx1];
        u_int16_t red2 = image[idx2];

        u_int16_t green1 = image[idx1 + npixels];
        u_int16_t green2 = image[idx2 + npixels];

        u_int16_t blue1 = image[idx1 + 2*npixels];
        u_int16_t blue2 = image[idx2 + 2*npixels];

        gray_image[idx1] = 0.299*red1 + 0.587*green1 + 0.114*blue1;
        gray_image[idx2] = 0.299*red2 + 0.587*green2 + 0.114*blue2;
    }
}

// calculating two pixels at a time to improve cache hit rate
__global__ void simple_threshold(u_int16_t *image, u_int16_t *output, u_int64_t npixels, u_int16_t threshold) {
    u_int64_t idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    idx1 *= 2;
    u_int64_t idx2 = idx1 + 1;

    if (idx2 < npixels) {
        u_int16_t val1 = image[idx1];
        u_int16_t val2 = image[idx2];

        output[idx1] = val1 > threshold ? val1 : 0;
        output[idx2] = val2 > threshold ? val2 : 0;
    }
}

__global__ void adaptiveThresholdingKernel(u_int16_t *image, u_int16_t *output, u_int64_t width, u_int64_t height, u_int16_t windowSize, u_int16_t offset) {
    u_int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    u_int64_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        
        // finestra quadrata centrata sul pixel (x, y)
        windowSize /= 2;
        u_int64_t startX = max(x - windowSize, 0L);
        u_int64_t endX = min(x + windowSize, width);
        u_int64_t startY = max(y - windowSize, 0L);
        u_int64_t endY = min(y + windowSize, height);

        // Calcola la media del blocco locale
        u_int64_t sum = 0; // max teorical window size is: sqrt(2^64 / 2^16) = sqrt(2^48) = *2^24* > 2^16)
        for (u_int64_t i = startY; i < endY; i++) {
            for (u_int64_t j = startX; j < endX; j++) {
                sum += image[i * width + j];
            }
        }
        u_int16_t localMean = sum / ((endX - startX) * (endY - startY));
        u_int16_t pixel = image[y * width + x];

        // Applica il thresholding adattivo
        output[y * width + x] = (pixel > (localMean + offset)) ? pixel : 0;
    }
}

// Ogni thread si occupa di un pixel dell'immagine ridotta.
__global__ void reduce_image(u_int16_t *image, u_int16_t *reduced_image, u_int64_t width, u_int64_t height, u_int16_t reduce_factor) {
    u_int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    u_int64_t y = blockIdx.y * blockDim.y + threadIdx.y;

    u_int64_t new_width = width / reduce_factor;
    u_int64_t new_height = height / reduce_factor;

    if (x < new_width && y < new_height) {
        u_int64_t sum = 0;
        u_int32_t out_of_range = 0;
        for (u_int16_t i = 0; i < reduce_factor; i++) {
            for (u_int16_t j = 0; j < reduce_factor; j++) {
                u_int64_t orig_x = x * reduce_factor + i;
                u_int64_t orig_y = y * reduce_factor + j;
                if (orig_x >= width || orig_y >= height) {
                    out_of_range++;
                    continue;
                }
                sum += image[orig_y * width + orig_x];
            }
        }
        reduced_image[y * new_width + x] = sum / (reduce_factor * reduce_factor - out_of_range);
    }
}

__global__ void adaptiveThresholdingApprossimative(
            u_int16_t *image, u_int16_t *output, u_int64_t width, u_int64_t height,
            u_int16_t *reduced_image, u_int16_t reduce_factor, u_int16_t windowSize, u_int16_t offset
    )
{
    u_int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    u_int64_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        
        // square windows centered in (x, y)
        windowSize /= 2;
        u_int64_t startX = max(x - windowSize, 0L) / reduce_factor;
        u_int64_t endX = min(x + windowSize, width) / reduce_factor;
        u_int64_t startY = max(y - windowSize, 0L) / reduce_factor;
        u_int64_t endY = min(y + windowSize, height) / reduce_factor;

        // local block mean
        u_int32_t sum = 0;
        u_int64_t reduced_width = width / reduce_factor;
        for (u_int64_t i = startY; i < endY; i++) {
            for (u_int64_t j = startX; j < endX; j++) {
                sum += reduced_image[i * reduced_width + j];
            }
        }
        u_int16_t localMean = sum / ((endX - startX) * (endY - startY));
        u_int16_t pixel = image[y * width + x];

        // apply adaptive thresholding
        output[y * width + x] = (pixel > (localMean + offset)) ? pixel : 0;
    }
}



// Kernel per la rilevazione delle stelle
__global__ void detect_stars(u_int16_t *input, u_int16_t *output, u_int64_t width, u_int64_t height, u_int16_t windowSize_star) {
    u_int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    u_int64_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    u_int64_t idx = y * width + x;
    u_int16_t current = input[idx];
    
    // Se il pixel corrente è nero, non è una stella
    if (current == 0)
        return;

    // Camminata a spirale
    bool is_star = true;
    bool all_black = true;   // una stella deve essere contenuta in un quadrato di tutti pixel neri

    int8_t directions[4][2] = {{1,0},{0,1},{-1,0},{0,-1}};
    u_int8_t dirIndex = 0;
    u_int32_t stepLimit = 1, stepCount = 0;
    u_int64_t current_idx;

    while(stepLimit < windowSize_star) {
        x += directions[dirIndex][0];
        y += directions[dirIndex][1];
        stepCount++;

        // Controlla se all’interno dell’immagine
        if(x >= width || y >= height) {
            is_star = false; 
            break;
        }

        // Controlla se il pixel corrente è maggiore del pixel centrale
        // se è maggiore allora non è una stella ed esco dal ciclo
        current_idx = y * width + x;
        if (input[current_idx] > current) {
            is_star = false;
            break;
        }
        
        // Controllo se non c'è un pixel candidato come centro con stessa luminosità
        // se esiste, allora controllo idx, se idx è maggiore di inizial_idx allora non è una stella 
        if (input[current_idx] == current) {
            if (current_idx > idx) {
                is_star = false;
                break;
            }
        }

        // Controllo se il pixel corrente è nero
        if (all_black && input[current_idx] > 0) {
            all_black = false;
        }

        // Controllo se ho completato un lato della spirale
        if(stepCount == stepLimit) {
            stepCount = 0;
            
            // Cambio direzione
            dirIndex++;
            // Controllo se ho completato un giro della spirale, allora resetto i contatori
            if (dirIndex % 4 == 0) {
                dirIndex = 0;
                // Controllo se tutti i pixel sono neri, allora finisco la ricerca della stella
                if (all_black) {
                    break;
                }
                all_black = true;
            }

            // Incremento il limite di passi ogni due cambi direzioni
            if(dirIndex % 2 == 0)
                stepLimit++; // equivale al doppio dei "giri" che ho fatto
        }
    }

    // Se è una stella, disegno il quadrato
    if (is_star && all_black && (stepLimit/2) > 2 && stepLimit < windowSize_star) {
        // (sx, sy) sono le coordinate dell'angolo in basso a sinistra del quadrato

        for (int i = x; i < x + stepLimit; i++) {
            // disegno il lato inferiore
            current_idx = y * width + i;
            output[current_idx] = 65535;

            // disegno il lato superiore
            current_idx = (y + stepLimit - 1) * width + i;
            output[current_idx] = 65535;
        }

        for (int j = y; j < y + stepLimit; j++) {
            // disegno il lato destro
            current_idx = j * width + x + stepLimit - 1;
            output[current_idx] = 65535;

            // disegno il lato sinistro
            current_idx = j * width + x;
            output[current_idx] = 65535;
        }
    }
}

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
__global__ void new_detect_stars(u_int16_t *input, u_int16_t *output, u_int64_t width, u_int64_t height, u_int16_t max_star_size, u_int16_t min_star_size, int x_debug, int y_debug) {
    // coordinates
    u_int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    u_int64_t y = blockIdx.y * blockDim.y + threadIdx.y;

    // boundary check
    if (x >= width || y >= height)
        return;

    // get the pixel index and the value
    u_int64_t idx = y * width + x;
    u_int16_t current = input[idx];

    bool is_debug = (x == x_debug && y == y_debug);
    if (is_debug) printf("Debug: current pixel value at idx %d is %u, x is %lu, y is %lu\n", idx, current, x, height - y);
    
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
        if (is_debug) printf("Debug: stepCurrentLimit[0] = %u, stepCurrentLimit[1] = %u, dir = %u, dir_x_or_y = %u, stepCount = %u, cx = %lu, cy = %lu\n", stepCurrentLimit[0], stepCurrentLimit[1], dir, dir_x_or_y, stepCount, x, height - y); 

        // Controllo se ho completato un lato
        if(stepCount == stepCurrentLimit[dir_x_or_y]) {
            stepCount = 0;

            if (is_debug) printf("Debug: completed side in direction %u, all_black = %d\n", dir, all_black);

            // Incremento il limite di passi della direzione corrente
            // solo se non ho finito la direzione precedente
            // es. se ho finito il lato in basso, non devo incrementare il limite 
            if (!finished_dir[previous_dir(dir)])
                stepCurrentLimit[dir_x_or_y]++;

            // Controllo se tutti i pixel sono neri, allora finisco la ricerca della stella nella direzione
            if (all_black && stepCurrentLimit[dir_x_or_y] > min_star_size) {
                finished_dir[dir] = true;

                // controllo se ho finito tutte le direzioni
                if (finished_dir[0] && finished_dir[1] && finished_dir[2] && finished_dir[3]) {
                    if (is_debug) printf("Debug: all directions finished, breaking loop\n");
                    break;
                }
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
            if (is_debug) printf("Debug: direction %u finished, skipping %u steps (x = %lu, y = %lu)\n", dir, stepCount, x, y);
            x += directions[dir][0] * stepCount;
            y += directions[dir][1] * stepCount;
            if (is_debug) printf("Debug: new position after skipping (x = %lu, y = %lu)\n", x, y);
            if (x >= width-1 || y >= height-1) {
                is_star = false;
                if (is_debug) printf("Debug: out of bounds after moving, breaking loop (x = %lu, y = %lu, stepCount = %u)\n", x, y, stepCount);
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
            if (is_debug) printf("Debug: out of bounds at edge, breaking loop\n");
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
            if (is_debug) printf("Debug: pixel (%lu, %lu) is brighter than center, breaking loop\n", x, height - y);
            break;
        }
        
        // Controllo se non c'è un pixel candidato come centro con stessa luminosità
        // se esiste, allora controllo idx, se idx è maggiore di inizial_idx allora non è il centro di una stella, esco dal ciclo 
        if (input[current_idx] == current) {
            if (current_idx > idx) {
                is_star = false;
                if (is_debug) printf("Debug: pixel (%lu, %lu) has same brightness but higher idx, breaking loop\n", x, height - y);
                break;
            }
        }

        // Controllo se il pixel corrente è nero, se non lo è imposto all_black a false
        if (all_black && input[current_idx] > 0)
            all_black = false;
    }
    if (is_debug) printf("Debug: finished loop, is_star = %d, all_black = %d, stepCurrentLimit[0] = %u, stepCurrentLimit[1] = %u, finished_dir = {%d, %d, %d, %d}\n", is_star, all_black, stepCurrentLimit[0], stepCurrentLimit[1], finished_dir[0], finished_dir[1], finished_dir[2], finished_dir[3]);

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
        output[y * width + x] = 65535; // segna il centro della stella (opzionale)
        printf("Star detected at (%lu, %lu) with size %u, idx= %lu\n", start_x, height - start_y, final_dim, idx);
        //if (idx == idxDebug) printf("star found\n");
        draw_rectangle(output, width, min_x, min_y, stepCurrentLimit[0], stepCurrentLimit[1]);
    }
}

#endif // CUDA_DEVICE_THRESHOLDING_H