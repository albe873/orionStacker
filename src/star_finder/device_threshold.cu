
// fits data is in planar format
// calculating two pixels at a time to improve cache hit rate
__global__ void kernel_grayscale_planar(const uint16_t *image, uint16_t *gray_image, const uint64_t npixels, const uint64_t npixels2) {
    const uint64_t idx1 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const auto idx2 = idx1 + 1;

    if (idx2 < npixels) {
        auto red1 = image[idx1];
        auto red2 = image[idx2];

        auto green1 = image[idx1 + npixels];
        auto green2 = image[idx2 + npixels];

        auto blue1 = image[idx1 + npixels2];
        auto blue2 = image[idx2 + npixels2];

        gray_image[idx1] = 0.299f*red1 + 0.587f*green1 + 0.114f*blue1;
        gray_image[idx2] = 0.299f*red2 + 0.587f*green2 + 0.114f*blue2;
    }
    else if (idx2 == npixels) {
        auto red1 = image[idx1];
        auto green1 = image[idx1 + npixels];
        auto blue1 = image[idx1 + npixels2];
        gray_image[idx1] = 0.299f*red1 + 0.587f*green1 + 0.114f*blue1;
    }
}

__global__ void kernel_grayscale_planar(const uint8_t *image, uint8_t *gray_image, const uint64_t npixels, const uint64_t npixels2) {
    const uint64_t idx1 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const auto idx2 = idx1 + 1;
    const auto idx3 = idx1 + 2;
    const auto idx4 = idx1 + 3;

    if (idx2 < npixels) {
        auto red1 = image[idx1];
        auto red2 = image[idx2];
        auto red3 = image[idx3];
        auto red4 = image[idx4];

        auto green1 = image[idx1 + npixels];
        auto green2 = image[idx2 + npixels];
        auto green3 = image[idx3 + npixels];
        auto green4 = image[idx4 + npixels];

        auto blue1 = image[idx1 + npixels2];
        auto blue2 = image[idx2 + npixels2];
        auto blue3 = image[idx3 + npixels2];
        auto blue4 = image[idx4 + npixels2];

        gray_image[idx1] = 0.299f*red1 + 0.587f*green1 + 0.114f*blue1;
        gray_image[idx2] = 0.299f*red2 + 0.587f*green2 + 0.114f*blue2;
        gray_image[idx3] = 0.299f*red3 + 0.587f*green3 + 0.114f*blue3;
        gray_image[idx4] = 0.299f*red4 + 0.587f*green4 + 0.114f*blue4;
    }
    else if (idx2 == npixels) {
        auto red1 = image[idx1];
        auto green1 = image[idx1 + npixels];
        auto blue1 = image[idx1 + 2*npixels];
        gray_image[idx1] = 0.299*red1 + 0.587*green1 + 0.114*blue1;
    }
    else if (idx3 == npixels) {
        auto red1 = image[idx1];
        auto red2 = image[idx2];
        auto green1 = image[idx1 + npixels];
        auto green2 = image[idx2 + npixels];
        auto blue1 = image[idx1 + npixels2];
        auto blue2 = image[idx2 + npixels2];
        gray_image[idx1] = 0.299*red1 + 0.587*green1 + 0.114*blue1;
        gray_image[idx2] = 0.299*red2 + 0.587*green2 + 0.114*blue2;
    }
    else if (idx4 == npixels) {
        auto red1 = image[idx1];
        auto red2 = image[idx2];
        auto red3 = image[idx3];
        auto green1 = image[idx1 + npixels];
        auto green2 = image[idx2 + npixels];
        auto green3 = image[idx3 + npixels];
        auto blue1 = image[idx1 + npixels2];
        auto blue2 = image[idx2 + npixels2];
        auto blue3 = image[idx3 + npixels2];
        gray_image[idx1] = 0.299*red1 + 0.587*green1 + 0.114*blue1;
        gray_image[idx2] = 0.299*red2 + 0.587*green2 + 0.114*blue2;
        gray_image[idx3] = 0.299*red3 + 0.587*green3 + 0.114*blue3;
    }
}

// calculating two pixels at a time to improve cache hit rate
__global__ void kernel_simple_threshold(const uint16_t *image, uint8_t *output, const uint64_t npixels, const uint16_t threshold) {
    uint64_t idx1 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    auto idx2 = idx1 + 1;

    if (idx2 < npixels) {
        auto pixel1 = image[idx1];
        auto pixel2 = image[idx2];

        output[idx1] = pixel1 > threshold ? (pixel1 / 256) : 0;
        output[idx2] = pixel2 > threshold ? (pixel2 / 256) : 0;
    }
    else if (idx2 == npixels) {
        auto pixel1 = image[idx1];
        output[idx1] = pixel1 > threshold ? (pixel1 / 256) : 0;
    }
}

__global__ void kernel_simple_threshold(const uint8_t *image, uint8_t *output, const uint64_t npixels, const uint8_t threshold) {
    const uint64_t idx1 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const auto idx2 = idx1 + 1;
    const auto idx3 = idx1 + 2;
    const auto idx4 = idx1 + 3;

    if (idx2 < npixels) {
        auto val1 = image[idx1];
        auto val2 = image[idx2];
        auto val3 = image[idx3];
        auto val4 = image[idx4];
        output[idx1] = val1 > threshold ? val1 : 0;
        output[idx2] = val2 > threshold ? val2 : 0;
        output[idx3] = val3 > threshold ? val3 : 0;
        output[idx4] = val4 > threshold ? val4 : 0;
    }
    else if (idx2 == npixels) {
        auto val1 = image[idx1];
        output[idx1] = val1 > threshold ? val1 : 0;
    }
    else if (idx3 == npixels) {
        auto val1 = image[idx1];
        auto val2 = image[idx2];
        output[idx1] = val1 > threshold ? val1 : 0;
        output[idx2] = val2 > threshold ? val2 : 0;
    }
    else if (idx4 == npixels) {
        auto val1 = image[idx1];
        auto val2 = image[idx2];
        auto val3 = image[idx3];
        output[idx1] = val1 > threshold ? val1 : 0;
        output[idx2] = val2 > threshold ? val2 : 0;
        output[idx3] = val3 > threshold ? val3 : 0;
    }
}

template <typename T>
__global__ void kernel_adaptive_threshold(const T *image, uint8_t *output, const uint64_t width, const uint64_t height, 
                                          uint16_t windowSize, const T offset) {
    uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        const uint64_t startX = (x > windowSize) ? x - windowSize : 0;
        const uint64_t endX = min((uint64_t)(x + windowSize), width - 1);
        const uint64_t startY = (y > windowSize) ? y - windowSize : 0;
        const uint64_t endY = min((uint64_t)(y + windowSize), height - 1);

        uint64_t sum = 0;
        for (uint64_t i = startY; i <= endY; i++) {
            for (uint64_t j = startX; j <= endX; j++) {
                sum += image[i * width + j];
            }
        }
        uint64_t num_pixels = (endX - startX + 1) * (endY - startY + 1);
        T localMean = (num_pixels > 0) ? (sum / num_pixels) : 0;
        T pixel = image[y * width + x];

        output[y * width + x] = (pixel > (localMean + offset)) ? pixel / 256 : 0;
    }
}


// Ogni thread si occupa di un pixel dell'immagine ridotta.
template<typename T>
__global__ void kernel_reduce_image(const T *image, T *reduced_image, uint64_t width, uint64_t height, 
                                    uint64_t new_width, uint64_t new_height, uint16_t reduce_factor, uint32_t squared_reduce_factor) {
    uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < new_width && y < new_height) {
        uint64_t sum = 0;
        uint32_t out_of_range = 0;
        uint64_t orig_x, orig_y;
        for (uint16_t i = 0; i < reduce_factor; i++) {
            for (uint16_t j = 0; j < reduce_factor; j++) {
                orig_x = x * reduce_factor + i;
                orig_y = y * reduce_factor + j;
                if (orig_x >= width || orig_y >= height) {
                    out_of_range++;
                    continue;
                }
                sum += image[orig_y * width + orig_x];
            }
        }
        reduced_image[y * new_width + x] = sum / (squared_reduce_factor - out_of_range);
    }
}

template <typename T>
__global__ void kernel_adaptive_threshold_approximate(
            const T *image, uint8_t *output, uint64_t width, uint64_t height,
            const T *reduced_image, uint64_t reduced_width, uint16_t reduce_factor, uint16_t windowSize, T offset
    ) 
{
    uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        uint16_t halfWindow = windowSize / 2;
        const uint64_t startX = (x > halfWindow) ? (x - halfWindow) / reduce_factor : 0;
        const uint64_t endX = min((uint64_t)(x + halfWindow), width - 1) / reduce_factor;
        const uint64_t startY = (y > halfWindow) ? (y - halfWindow) / reduce_factor : 0;
        const uint64_t endY = min((uint64_t)(y + halfWindow), height - 1) / reduce_factor;

        uint64_t sum = 0;
        for (uint64_t i = startY; i <= endY; i++) {
            for (uint64_t j = startX; j <= endX; j++) {
                sum += reduced_image[i * reduced_width + j];
            }
        }

        uint64_t num_pixels = (endX - startX + 1) * (endY - startY + 1);
        T localMean = (num_pixels > 0) ? (sum / num_pixels) : 0;
        T pixel = image[y * width + x];

        output[y * width + x] = (pixel > (localMean + offset)) ? pixel / 256 : 0;
    }
}
