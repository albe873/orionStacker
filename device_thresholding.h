#ifndef CUDA_DEVICE_THRESHOLDING_H
#define CUDA_DEVICE_THRESHOLDING_H

// fits data is in planar format
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


__global__ void simple_threshold(u_int16_t *image, u_int16_t *output, u_int64_t npixels, u_int16_t threshold) {
    u_int64_t idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    idx1 *= 2;
    u_int64_t idx2 = idx1 + 1;

    if (idx2 < npixels) {
        u_int16_t val1 = image[idx1];
        u_int16_t val2 = image[idx2];

        output[idx1] = val1 > threshold ? 65535 : 0;
        output[idx2] = val2 > threshold ? 65535 : 0;
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
        u_int32_t sum = 0;
        for (u_int64_t i = startY; i < endY; i++) {
            for (u_int64_t j = startX; j < endX; j++) {
                sum += image[i * width + j];
            }
        }
        u_int16_t localMean = sum / ((endX - startX) * (endY - startY));
        u_int16_t pixel = image[y * width + x];

        // Applica il thresholding adattivo
        output[y * width + x] = (pixel > (localMean + offset)) ? 65535 : 0;
    }
}

// Ogni thread si occupa di un pixel dell'immagine ridotta.
__global__ void reduce_image(u_int16_t *image, u_int16_t *reduced_image, u_int64_t width, u_int64_t height, u_int16_t reduce_factor) {
    u_int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    u_int64_t y = blockIdx.y * blockDim.y + threadIdx.y;

    u_int64_t new_width = width / reduce_factor;
    u_int64_t new_height = height / reduce_factor;

    if (x < new_width && y < new_height) {
        u_int32_t sum = 0;
        for (u_int32_t i = 0; i < reduce_factor; i++) {
            for (u_int32_t j = 0; j < reduce_factor; j++) {
                u_int32_t orig_x = x * reduce_factor + i;
                u_int32_t orig_y = y * reduce_factor + j;
                if (orig_x >= width || orig_y >= height) {
                    continue;
                }
                sum += image[orig_y * width + orig_x];
            }
        }
        reduced_image[y * new_width + x] = sum / (reduce_factor * reduce_factor);
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
        
        // finestra quadrata centrata sul pixel (x, y)
        windowSize /= 2;
        u_int64_t startX = max(x - windowSize, 0L) / reduce_factor;
        u_int64_t endX = min(x + windowSize, width) / reduce_factor;
        u_int64_t startY = max(y - windowSize, 0L) / reduce_factor;
        u_int64_t endY = min(y + windowSize, height) / reduce_factor;

        // Calcola la media del blocco locale
        u_int32_t sum = 0;
        u_int64_t reduced_width = width / reduce_factor;
        for (u_int64_t i = startY; i < endY; i++) {
            for (u_int64_t j = startX; j < endX; j++) {
                sum += reduced_image[i * reduced_width + j];
            }
        }
        u_int16_t localMean = sum / ((endX - startX) * (endY - startY));
        u_int16_t pixel = image[y * width + x];

        // Applica il thresholding adattivo
        output[y * width + x] = (pixel > (localMean + offset)) ? 65535 : 0;
    }
}



// Kernel per la rilevazione delle stelle
__global__ void detect_stars(u_int16_t *input, u_int16_t *output, u_int64_t width, u_int64_t height, u_int16_t windowSize_star) {
    u_int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    u_int64_t y = blockIdx.y * blockDim.y + threadIdx.y;
    u_int16_t halfWindow = windowSize_star / 2;

    if (x >= width || y >= height) 
        return;

    u_int64_t idx = y * width + x;
    u_int16_t current = input[idx];

    // Camminata a spirale
    bool is_star = true;
    bool allBlack = true;   // una stella deve essere circondata da pixel neri
    u_int32_t spiralSteps = (2 * halfWindow + 1) * (2 * halfWindow + 1) - 1;
    int8_t directions[4][2] = {{1,0},{0,1},{-1,0},{0,-1}};
    u_int32_t stepLimit = 1, stepCount = 0;
    u_int8_t dirIndex = 0;
    u_int64_t sx = x, sy = y;

    u_int64_t inizitial_idx = y * width + x;
    u_int64_t current_idx;

    for(int s = 1; s <= spiralSteps; s++) {
        sx += directions[dirIndex][0];
        sy += directions[dirIndex][1];
        stepCount++;

        // Controlla se all’interno dell’immagine
        if(sx < 0 || sx >= width || sy < 0 || sy >= height) {
            is_star = false; 
            break;
        }

        // Controlla se il pixel corrente è maggiore del pixel centrale
        // se è maggiore allora non è una stella ed esco dal ciclo
        current_idx = sy * width + sx;
        if (input[current_idx] > current) {
            is_star = false;
            break;
        }
        
        // Controllo se non c'è un pixel candidato come centro con stessa luminosità
        // se esiste, allora controllo idx, se idx è maggiore di inizial_idx allora non è una stella 
        if (input[current_idx] == current) {
            if (current_idx > inizitial_idx) {
                is_star = false;
                break;
            }
        }

        // Controllo se il pixel corrente è nero
        if (allBlack && input[current_idx] > 0) {
            allBlack = false;
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
                if (allBlack) {
                    break;
                }
                allBlack = true;
            }

            // Incremento il limite di passi ogni due cambi direzioni
            if(dirIndex % 2 == 0)
                stepLimit++;
        }
    }

    // Se è una stella
    if (is_star && allBlack) {
        output[idx] = 65535;
    }
}


#endif // CUDA_DEVICE_THRESHOLDING_H