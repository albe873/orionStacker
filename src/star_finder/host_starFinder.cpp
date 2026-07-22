#ifndef CPU_STARFINDER
#define CPU_STARFINDER

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "star_finder.h"
#include "host_otsu_centralized.cpp"

void to_grayscale_planar(const uint16_t* __restrict__ image, uint16_t* __restrict__ gray_image, uint64_t npixels) {
    #pragma omp parallel for
    for(uint64_t i = 0; i < npixels; i++) {
        uint16_t red = image[i];
        uint16_t green = image[i + npixels];
        uint16_t blue = image[i + 2*npixels];
        gray_image[i] = 0.299*red + 0.587*green + 0.114*blue;
    }
}

void simple_threshold(const uint16_t* __restrict__ image, uint8_t* __restrict__ output, uint64_t npixels, uint16_t threshold) {
    #pragma omp parallel for
    for(uint64_t i = 0; i < npixels; i++) {
        output[i] = image[i] > threshold ? image[i] / 256 : 0;
    }
}

void adaptive_threshold(const uint16_t* __restrict__ image, uint8_t* __restrict__ output, uint64_t width, uint64_t height, 
                        uint16_t windowSize, uint16_t offset) {
    windowSize /= 2;
    #pragma omp parallel for
    for(uint64_t y = 0; y < height; y++) {
        //printf("Processing y %ld\n", y);
        for(uint64_t x = 0; x < width; x++) {
            uint64_t startX = x > windowSize          ? x - windowSize : 0;
            uint64_t endX   = x + windowSize < width  ? x + windowSize : width;
            uint64_t startY = y > windowSize          ? y - windowSize : 0;
            uint64_t endY   = y + windowSize < height ? y + windowSize : height;

            uint32_t sum = 0;
            for(uint64_t i = startY; i < endY; i++) {
                for(uint64_t j = startX; j < endX; j++) {
                    sum += image[i * width + j];
                }
            }
            uint16_t localMean = sum / ((endX - startX) * (endY - startY));
            uint16_t pixel = image[y * width + x];
            output[y * width + x] = (pixel > (localMean + offset)) ? pixel / 256 : 0;
        }
    }
}

void reduce_image(const uint16_t* __restrict__ image, uint16_t* __restrict__ reduced_image, uint64_t width, uint64_t height, 
                  uint16_t reduce_factor) {
    uint64_t new_width = width / reduce_factor;
    uint64_t new_height = height / reduce_factor;
    
    #pragma omp parallel for
    for(uint64_t y = 0; y < new_height; y++) {
        for(uint64_t x = 0; x < new_width; x++) {
            uint32_t sum = 0;
            for(uint32_t i = 0; i < reduce_factor; i++) {
                for(uint32_t j = 0; j < reduce_factor; j++) {
                    uint32_t orig_x = x * reduce_factor + i;
                    uint32_t orig_y = y * reduce_factor + j;
                    if(orig_x >= width || orig_y >= height)
                        continue;
                    sum += image[orig_y * width + orig_x];
                }
            }
            reduced_image[y * new_width + x] = sum / (reduce_factor * reduce_factor);
        }
    }
}

void adaptive_threshold_approximate(const uint16_t* __restrict__ image, uint8_t* __restrict__ output, uint64_t width, 
                                    uint64_t height, uint16_t* __restrict__ reduced_image, 
                                    uint16_t reduce_factor, uint16_t windowSize, 
                                    uint16_t offset) {
    windowSize /= 2;
    #pragma omp parallel for
    for(uint64_t y = 0; y < height; y++) {
        for(uint64_t x = 0; x < width; x++) {
            uint64_t startX = (x > windowSize ? x - windowSize : 0) / reduce_factor;
            uint64_t endX = (x + windowSize < width ? x + windowSize : width) / reduce_factor;
            uint64_t startY = (y > windowSize ? y - windowSize : 0) / reduce_factor;
            uint64_t endY = (y + windowSize < height ? y + windowSize : height) / reduce_factor;

            uint32_t sum = 0;
            uint64_t reduced_width = width / reduce_factor;
            for(uint64_t i = startY; i < endY; i++) {
                for(uint64_t j = startX; j < endX; j++) {
                    sum += reduced_image[i * reduced_width + j];
                }
            }
            uint16_t localMean = sum / ((endX - startX) * (endY - startY));
            uint16_t pixel = image[y * width + x];
            output[y * width + x] = (pixel > (localMean + offset)) ? pixel / 256 : 0;
        }
    }
}

void draw_stars(u_int16_t* __restrict__ img, u_int64_t width, const star *stars, u_int32_t n_stars) {
    for(uint32_t i = 0; i < n_stars; i++) {
        star s = stars[i];

        // ciclo sui lati orizzontali
        u_int64_t idx1 = s.start_y * width + s.start_x;
        u_int64_t idx2 = (s.start_y + s.size_y) * width + s.start_x;
        for (int i = 0; i < s.size_x; i++) {
            img[idx1] = 65535;
            img[idx2] = 65535;
            // basta un incremento unitario
            idx1++;
            idx2++;
        }

        // ciclo sui lati verticali
        idx1 = s.start_y * width + s.start_x;
        idx2 = s.start_y * width + s.start_x + s.size_x;
        for (int j = 0; j < s.size_y; j++) {
            img[idx1] = 65535;
            img[idx2] = 65535;
            // devo cambiare riga, incremento pari alla larghezza della riga
            idx1 += width;
            idx2 += width;
        }
        img[idx2] = 65535; // segna l'angolo in alto a destra

    }
}

// ---- Helper functions for star detection ----

inline int previous_dir(int dir) {
    return (dir == 0) ? 3 : dir - 1;
}

inline void change_direction(int8_t &dir, int8_t &dir_x_or_y) {
    // cambio direzione x o y (inverto sempre)
    dir_x_or_y = 1 - dir_x_or_y;

    // aumento dir, poi se ho completato un giro resetto i contatori
    dir++;
    if (dir == 4)
        dir = 0;
}

inline void write_star(star *stars, u_int32_t &num_stars, u_int32_t max_stars, 
                                  uint64_t start_x, uint64_t start_y,
                                  uint32_t size_x,  uint32_t size_y) {
    u_int32_t idx = num_stars;
    num_stars++;
    if (idx < max_stars) {
        stars[idx].start_x = start_x;
        stars[idx].start_y = start_y;
        stars[idx].size_x = size_x;
        stars[idx].size_y = size_y;
    }
}

void detect_stars(const uint8_t* __restrict__ input, uint64_t width, uint64_t height, 
                  uint16_t max_star_size, uint16_t min_star_size,
                  star *stars, uint32_t &num_stars, uint32_t max_stars) {
    num_stars = 0;
    const int8_t directions[4][2] = {{1,0},{0,1},{-1,0},{0,-1}};  // variazione x e y per ogni direzione

    #pragma omp parallel for
    for(uint64_t y = 0; y < height; y++) {
        for(uint64_t x = 0; x < width; x++) {
            uint64_t idx = y * width + x;
            auto current = input[idx];

            // Se il pixel corrente è nero, non è una stella
            if(current == 0)
                continue;

            bool is_star = true;
            bool all_black = true;                                  // una stella deve essere contenuta in un quadrato di pixel neri
            bool finished_dir[4] = {false, false, false, false};    // per ogni direzione, indica se ho finito di esplorare quella direzione

            int32_t   stepCount = 0;                              // contatore dei passi fatti nella direzione corrente
            int32_t   stepCurrentLimit[2] = {1, 1};               // step limit per x e y
            int8_t    dir = 0;                                    // direzione corrente
            int8_t    dir_x_or_y = 0;                             // = 0 se x, = 1 se y, per comodità

            u_int64_t current_idx;                                // indice del pixel corrente

            // Use local variables for star detection to avoid modifying loop variables
            uint64_t cur_x = x;
            uint64_t cur_y = y;
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
                    cur_x += directions[dir][0] * stepCount;
                    cur_y += directions[dir][1] * stepCount;
                    if (cur_x >= width-1 || cur_y >= height-1) {
                        is_star = false;
                        break;
                    }

                    continue;
                }

                // mi muovo di un passo
                cur_x += directions[dir][0];
                cur_y += directions[dir][1];
                stepCount++;

                // check se sono ai bordi dell'immagine
                if (cur_x >= width || cur_y >= height) {
                    is_star = false;
                    break;
                }     
                
                // aggiorno le coordinate minime
                min_x = cur_x < min_x ? cur_x : min_x;
                min_y = cur_y < min_y ? cur_y : min_y;

                // Controlla se il pixel corrente è maggiore del pixel centrale
                // se è maggiore allora non è il centro di una stella, esco dal ciclo
                current_idx = cur_y * width + cur_x;
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

            } // while

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

        } // for x
    } // for y
}


void compute_threshold( const u_int16_t* __restrict__ img, u_int8_t* __restrict__ out_img,
                        u_int64_t width, u_int64_t height, 
                        threshold_params params) {
    u_int64_t npixels = width * height;
    // in case of fast adaptive thresholding, allocate reduced image
    u_int16_t *reduced_image = nullptr;
    if (params.type == TR_FAST_ADAPTIVE) {
        reduced_image = (u_int16_t *)malloc((npixels / params.reduce_factor / params.reduce_factor) * sizeof(u_int16_t));
        if (reduced_image == nullptr) {
            fprintf(stderr, "Error allocating memory for reduced image\n");
            exit(EXIT_FAILURE);
        }
    }

    // --- Apply thresholding ---
    switch (params.type) {
        case TR_SIMPLE:
            simple_threshold(img, out_img, npixels, params.threshold);
            break;
        case TR_ADAPTIVE:
            adaptive_threshold(img, out_img, width, height, params.window_size, params.threshold);
            break;
        case TR_FAST_ADAPTIVE:
            reduce_image(img, reduced_image, width, height, params.reduce_factor);
            adaptive_threshold_approximate(img, out_img, width, height, reduced_image,
                params.reduce_factor, params.window_size, params.threshold);
            break;
        case OTSU_CENTRALIZED:
            cpu_otsu_centralized_threshold(img, out_img, width, height, params.window_size, params.threshold_scale);
            break;
    }
    
    // free allocated memory
    if (params.type == TR_FAST_ADAPTIVE) {
        free(reduced_image);
    }
}


void sum_brightness_planar(const u_int16_t* __restrict__ input_rgb, u_int64_t input_width, u_int64_t npixels, star s, star_detail *sd) {
    uint64_t idx;
    for (uint64_t x = s.start_x; x < s.start_x + s.size_x; x++) {
        for (uint64_t y = s.start_y; y < s.start_y + s.size_y; y++) {
            idx = y * input_width + x;
            sd->b_red   += (double) input_rgb[idx];
            idx += npixels;
            sd->b_green += (double) input_rgb[idx];
            idx += npixels;
            sd->b_blue  += (double) input_rgb[idx];
        }
    }
    sd->b = 0.299*sd->b_red + 0.587*sd->b_green + 0.114*sd->b_blue;
}


void baricenter_uint16_host(const u_int16_t* __restrict__ input_grayscale, u_int64_t input_width, star s, star_detail *sd) {
    uint64_t idx;
    for (uint64_t x = s.start_x; x < s.start_x + s.size_x; x++) {
        for (uint64_t y = s.start_y; y < s.start_y + s.size_y; y++) {
            idx = y * input_width + x;
            uint16_t value = input_grayscale[idx];
            sd->x += (double) x * value / sd->b;
            sd->y += (double) y * value / sd->b;
        }
    }
}



void populate_star_details(star_detail *stars_details, star *stars, u_int32_t n_stars, 
                           const u_int16_t* __restrict__ img_rgb, const u_int16_t* __restrict__ img_gray,
                           u_int64_t width, u_int64_t npixels) {
    // for each star, first I compute brightness then the baricenter
    // (I need the brightness sums to compute the baricenter)
    #pragma omp parallel for
    for (u_int32_t i = 0; i < n_stars; i++) {
        init_star_detail(&stars_details[i]);
        sum_brightness_planar(img_rgb, width, npixels, stars[i], &stars_details[i]);
    } 

    #pragma omp parallel for
    for (u_int32_t i = 0; i < n_stars; i++) {
        baricenter_uint16_host(img_gray, width, stars[i], &stars_details[i]);
    }
}

#endif // CPU_STARFINDER_H