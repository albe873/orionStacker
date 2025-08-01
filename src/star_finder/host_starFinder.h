#ifndef CPU_STARFINDER_H
#define CPU_STARFINDER_H

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

void to_grayscale_fits_cpu(uint16_t *image, uint16_t *gray_image, uint64_t npixels) {
    for(uint64_t i = 0; i < npixels; i++) {
        uint16_t red = image[i];
        uint16_t green = image[i + npixels];
        uint16_t blue = image[i + 2*npixels];
        gray_image[i] = 0.299*red + 0.587*green + 0.114*blue;
    }
}

void simple_threshold_cpu(uint16_t *image, uint16_t *output, uint64_t npixels, uint16_t threshold) {
    for(uint64_t i = 0; i < npixels; i++) {
        output[i] = image[i] > threshold ? image[i] : 0;
    }
}

void adaptiveThresholding_cpu(uint16_t *image, uint16_t *output, uint64_t width, uint64_t height, 
                            uint16_t windowSize, uint16_t offset) {
    windowSize /= 2;
    for(uint64_t y = 0; y < height; y++) {
        //printf("Processing y %ld\n", y);
        for(uint64_t x = 0; x < width; x++) {
            uint64_t startX = x > windowSize ? x - windowSize : 0;
            uint64_t endX = x + windowSize < width ? x + windowSize : width;
            uint64_t startY = y > windowSize ? y - windowSize : 0;
            uint64_t endY = y + windowSize < height ? y + windowSize : height;

            uint32_t sum = 0;
            for(uint64_t i = startY; i < endY; i++) {
                for(uint64_t j = startX; j < endX; j++) {
                    sum += image[i * width + j];
                }
            }
            uint16_t localMean = sum / ((endX - startX) * (endY - startY));
            uint16_t pixel = image[y * width + x];
            output[y * width + x] = (pixel > (localMean + offset)) ? pixel : 0;
        }
    }
}

void reduce_image_cpu(uint16_t *image, uint16_t *reduced_image, uint64_t width, uint64_t height, 
                     uint16_t reduce_factor) {
    uint64_t new_width = width / reduce_factor;
    uint64_t new_height = height / reduce_factor;

    for(uint64_t y = 0; y < new_height; y++) {
        for(uint64_t x = 0; x < new_width; x++) {
            uint32_t sum = 0;
            for(uint32_t i = 0; i < reduce_factor; i++) {
                for(uint32_t j = 0; j < reduce_factor; j++) {
                    uint32_t orig_x = x * reduce_factor + i;
                    uint32_t orig_y = y * reduce_factor + j;
                    if(orig_x >= width || orig_y >= height) continue;
                    sum += image[orig_y * width + orig_x];
                }
            }
            reduced_image[y * new_width + x] = sum / (reduce_factor * reduce_factor);
        }
    }
}

void adaptiveThresholdingApprossimative_cpu(uint16_t *image, uint16_t *output, uint64_t width, 
                                          uint64_t height, uint16_t *reduced_image, 
                                          uint16_t reduce_factor, uint16_t windowSize, 
                                          uint16_t offset) {
    windowSize /= 2;
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
            output[y * width + x] = (pixel > (localMean + offset)) ? pixel : 0;
        }
    }
}

void detect_stars_cpu(uint16_t *input, uint16_t *output, uint64_t width, uint64_t height, 
                     uint16_t windowSize_star) {
    for(uint64_t y = 0; y < height; y++) {
        for(uint64_t x = 0; x < width; x++) {
            uint64_t idx = y * width + x;
            uint16_t current = input[idx];

            bool is_star = true;
            bool allBlack = true;

            int8_t directions[4][2] = {{1,0},{0,1},{-1,0},{0,-1}};
            uint8_t dirIndex = 0;
            uint32_t stepLimit = 1, stepCount = 0;
            int64_t curr_x = x, curr_y = y;
            uint64_t current_idx;

            while(stepLimit < windowSize_star) {
                curr_x += directions[dirIndex][0];
                curr_y += directions[dirIndex][1];
                stepCount++;

                if(curr_x < 0 || curr_x >= width || curr_y < 0 || curr_y >= height) {
                    is_star = false;
                    break;
                }

                current_idx = curr_y * width + curr_x;
                if(input[current_idx] > current) {
                    is_star = false;
                    break;
                }

                if(input[current_idx] == current && current_idx > idx) {
                    is_star = false;
                    break;
                }

                if(allBlack && input[current_idx] > 0) {
                    allBlack = false;
                }

                if(stepCount == stepLimit) {
                    stepCount = 0;
                    dirIndex++;
                    if(dirIndex % 4 == 0) {
                        dirIndex = 0;
                        if(allBlack) break;
                        allBlack = true;
                    }
                    if(dirIndex % 2 == 0) stepLimit++;
                }
            }

            if(is_star && allBlack && (stepLimit/2) > 2 && stepLimit < windowSize_star) {
                for(int i = curr_x; i < curr_x + stepLimit; i++) {
                    current_idx = curr_y * width + i;
                    output[current_idx] = 65535;
                    current_idx = (curr_y + stepLimit - 1) * width + i;
                    output[current_idx] = 65535;
                }

                for(int j = curr_y; j < curr_y + stepLimit; j++) {
                    current_idx = j * width + curr_x + stepLimit - 1;
                    output[current_idx] = 65535;
                    current_idx = j * width + curr_x;
                    output[current_idx] = 65535;
                }
            }
        }
    }
}

#endif // CPU_STARFINDER_H