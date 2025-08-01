#ifndef CUDA_DEVICE_THRESHOLDING_H
#define CUDA_DEVICE_THRESHOLDING_H

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



// Kernel per la rilevazione delle stelle
__global__ void new_detect_stars(u_int16_t *input, u_int16_t *output, u_int64_t width, u_int64_t height, u_int16_t windowSize_star) {
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
    bool all_black[4] = {true, true, true, true};   // una stella deve essere contenuta in un quadrato di tutti pixel neri
    bool finished_dir[4] = {false, false, false, false};

    int8_t directions[4][2] = {{1,0},{0,1},{-1,0},{0,-1}};
    u_int8_t dirIndex = 0;
    u_int32_t stepCount = 0;
    u_int32_t stepLimit[2] = {1, 1};    // step limit per x e y
    int8_t dir_x_or_y = 0; // = 0 se x, = 1 se y
    u_int64_t current_idx;

    u_int64_t start_x = x;
    u_int64_t start_y = y;

    u_int32_t final_limit_x = 0;
    u_int32_t final_limit_y = 0;

    while(stepLimit[0] < windowSize_star && stepLimit[1] < windowSize_star) {

        // Controllo se ho completato un lato della spirale
        if(stepCount == stepLimit[dir_x_or_y]) {
            stepCount = 0;

            // Incremento il limite di passi della direzione corrente
            // solo se non ho finito la direzione
            if (!finished_dir[dirIndex]) {
                stepLimit[dir_x_or_y]++;

                // Controllo se tutti i pixel sono neri, allora finisco la ricerca della stella nella direzione
                if (all_black[dirIndex]) {
                    finished_dir[dirIndex] = true;

                    // controllo se ho finito tutte le direzioni
                    if (finished_dir[0] && finished_dir[1] && finished_dir[2] && finished_dir[3])
                        break;
                }
            }


            // cambio direzione
            dir_x_or_y = 1 - dir_x_or_y; // cambio da x a y o viceversa
            dirIndex++;
            
            // Controllo se ho completato un giro, allora resetto i contatori
            if (dirIndex % 4 == 0)
                dirIndex = 0;
            
            all_black[dirIndex] = true;
        }

        // se i pixel del lato corrente sono neri, allora non esploro più in quella direzione e passo alla sucessiva
        if (finished_dir[dirIndex]) {
            // imposto di quanto mi devo muovere 
            stepCount = stepLimit[dir_x_or_y];

            // mi muovo, saltando tutti i controlli e vado al ciclo sucessivo
            x += directions[dirIndex][0] * stepCount;
            y += directions[dirIndex][1] * stepCount;
            continue;
        }

        // mi muovo di un passo
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
        if (all_black[dirIndex] && input[current_idx] > 0) {
            all_black[dirIndex] = false;
        }
    }

    final_limit_x = stepLimit[0];
    final_limit_y = stepLimit[1];

    // Condizione finale: stella valida E tutte le direzioni terminate con nero
    bool surrounded_by_black = finished_dir[0] && finished_dir[1] && finished_dir[2] && finished_dir[3];

    // Usa il MINIMO dei limiti finali per definire la dimensione del quadrato da disegnare
    // e per il controllo della dimensione minima.
    // Assicurati che final_limit_x e final_limit_y siano stati salvati nel loop
    u_int32_t final_limit = min(final_limit_x, final_limit_y); // final_limit_x/y should be saved when loop breaks

    // Verifica le condizioni per disegnare
    if (is_star && surrounded_by_black && (final_limit / 2) > 2 && final_limit < windowSize_star) {
        // Disegna il quadrato basato sul centro INIZIALE e il limite MINIMO trovato
        int64_t limit = final_limit/2; // Usiamo int64_t per coerenza con le coordinate box_...

        // Calcola le coordinate del bounding box, clampate ai limiti dell'immagine
        // Il raggio è 'limit', quindi il box va da centro-limit a centro+limit
        // Usa initial_x/y (rinominati da start_x/y per chiarezza)
        int64_t box_start_x = max(0L, (int64_t)start_x - limit);
        int64_t box_start_y = max(0L, (int64_t)start_y - limit);
        int64_t box_end_x   = min((int64_t)width - 1, (int64_t)start_x + limit);
        int64_t box_end_y   = min((int64_t)height - 1, (int64_t)start_y + limit);

        // Disegna il bordo del quadrato (assicurati che le coordinate siano valide)
        // Lato superiore
        if (box_start_y >= 0 && box_start_y < height) { // Check row validity
            for (int64_t i = box_start_x; i <= box_end_x; ++i) {
                 if (i >= 0 && i < width) { // Check column validity
                    output[box_start_y * width + i] = 65535;
                 }
            }
        }
        // Lato inferiore (evita ridisegno se altezza è 1)
        if (box_end_y >= 0 && box_end_y < height && box_end_y != box_start_y) { // Check row validity
            for (int64_t i = box_start_x; i <= box_end_x; ++i) {
                 if (i >= 0 && i < width) { // Check column validity
                    output[box_end_y * width + i] = 65535;
                 }
            }
        }
        // Lato sinistro (escludi angoli già disegnati)
        if (box_start_x >= 0 && box_start_x < width) { // Check column validity
            for (int64_t j = box_start_y + 1; j < box_end_y; ++j) { // Start from +1 and end before end_y
                 if (j >= 0 && j < height) { // Check row validity
                    output[j * width + box_start_x] = 65535;
                 }
            }
        }
        // Lato destro (escludi angoli, evita ridisegno se larghezza è 1)
        if (box_end_x >= 0 && box_end_x < width && box_end_x != box_start_x) { // Check column validity
            for (int64_t j = box_start_y + 1; j < box_end_y; ++j) { // Start from +1 and end before end_y
                 if (j >= 0 && j < height) { // Check row validity
                    output[j * width + box_end_x] = 65535;
                 }
            }
        }
    }
}

#endif // CUDA_DEVICE_THRESHOLDING_H