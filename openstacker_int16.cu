#include "cuda_runtime.h"
#include <stdio.h>
#include <fitsio.h>
#include <dirent.h>
#include <string.h>
#include <stdint.h>

// Include STB image libraries
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


/*  --- AMD HIPify tool & HCC compiler ---

--  to generate hipified code, use the following command
hipify-clang openstacker_int16.cu --cuda-path=/opt/cuda

--  to compile the hipified code, use the following command
hipcc openstacker_int16.cu.hip  -o openstacker_int16 -lcfitsio -O3 -Wall
*/

#define CHECK(err) do { cuda_check((err), __FILE__, __LINE__); } while(false)
inline void cuda_check(cudaError_t error_code, const char *file, int line) {
    if (error_code != cudaSuccess) {
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
        fflush(stderr);
        exit(error_code);
    }
}

void rgbToGrayCPU(unsigned char *rgb, unsigned char *gray, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int rgbOffset = (y * width + x) * 3;
            int grayOffset = y * width + x;
            unsigned char r = rgb[rgbOffset];
            unsigned char g = rgb[rgbOffset + 1];
            unsigned char b = rgb[rgbOffset + 2];
            gray[grayOffset] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
        }
    }
}

__global__ void rgbToGrayGPU(unsigned char *d_rgb, unsigned char *d_gray, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        int grayIdx = y * width + x;
        d_gray[grayIdx] = (unsigned char)(0.299f * d_rgb[idx] + 0.587f * d_rgb[idx+1] + 0.114f * d_rgb[idx+2]);
    }
}

void process_fits_image(char *file_path, char *file_name) {
    fitsfile *fptr;
    int status = 0; // variabile di stato che CFITSIO aggiorna per rilevare eventuali errori

    printf("image to open: %s\n", file_path);

    // 1 APERTURA DEL FILE FITS
    if (fits_open_file(&fptr, file_path, READONLY, &status)) {
        fits_report_error(stderr, status);
        fprintf(stderr, "Could not open input file\n");
        return;
    }

    // 2 LETTURA DELLE DIMENSIONI DELL'IMMAGINE
    int naxis;                  // numero di assi dell’immagine (di solito 2 per immagini 2D)
    long naxes[2] = {1, 1};     // array che contiene la dimensione dell’immagine in larghezza e altezza
    
    // fits_get_img_dim e fits_get_img_size ottengono rispettivamente il numero di assi e la dimensione dell’immagine 
    if (fits_get_img_dim(fptr, &naxis, &status) || fits_get_img_size(fptr, 2, naxes, &status)) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        return;
    }
    int width = naxes[0];
    int height = naxes[1];
    size_t npixels = width * height; // tot pixel dell'immagine

    printf("Processing image: %s\n", file_path);
    printf("Image size: %dx%d\n", width, height);

    ///*3 ALLOCAZIONE MEMORIA PER IMMAGINE RGB E SCALA DI GRIGI
    uint8_t *rgb = (uint8_t *) malloc(npixels * 3 * sizeof(uint8_t));
    uint8_t *gray = (uint8_t *) malloc(npixels * sizeof(uint8_t));

    ///*4 LETTURA DATI IMMAGINE 
    uint16_t *fits_data = (uint16_t *) malloc(npixels * sizeof(uint16_t));

    // lettura dei dati dell’immagine FITS e memorizzazione in 'fits_data'
    if (fits_read_img(fptr, TUSHORT, 1, npixels, NULL, fits_data, NULL, &status)) {
        fits_report_error(stderr, status);
        fprintf(stderr, "Could not read input file\n");
        fits_close_file(fptr, &status);
        free(fits_data);
        return;
    }
    fits_close_file(fptr, &status);

    ///*5 CONVERSIONI ***********************************************************************/
    // Conversione dati FITS in formato RGB
    for (size_t i = 0; i < npixels; i++) {
        uint8_t value = (uint8_t)((float)(fits_data[i]) / (65535 / 255));
        rgb[i * 3] = value;
        rgb[i * 3 + 1] = value;
        rgb[i * 3 + 2] = value;
    }
    free(fits_data);


    // Conversione in scala di grigi su CPU
    rgbToGrayCPU(rgb, gray, width, height);
    char gray_file_name[1024];
    snprintf(gray_file_name, sizeof(gray_file_name), "output/%s", file_name);
    // Rimuove l'estensione del file
    char *dot = strrchr(gray_file_name, '.');
    if (dot) {
        *dot = '\0';
    }
    strcat(gray_file_name, ".png");
    stbi_write_png(gray_file_name, width, height, 1, gray, width);


    ///*6 ALLOCAZIONE MEMORIA SU  DEVICE *******************************************************/
    /*
    unsigned char *d_rgb, *d_gray;                          // copie GPU delle immagini rgb e gray
    size_t rgbSize = npixels * 3 * sizeof(unsigned char);
    size_t graySize = npixels * sizeof(unsigned char);
    CHECK(cudaMalloc((void **)&d_rgb, rgbSize));
    CHECK(cudaMalloc((void **)&d_gray, graySize));

    //7 LANCIO KERNEL ***********************************************************************
    // Copia dati su device
    CHECK(cudaMemcpy(d_rgb, rgb, rgbSize, cudaMemcpyHostToDevice));

    // Configurazione e lancio kernel CUDA
    dim3 block(8, 8);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    rgbToGrayGPU<<<grid, block>>>(d_rgb, d_gray, width, height);
    CHECK(cudaDeviceSynchronize());

    // Copia risultato dal device all'host
    CHECK(cudaMemcpy(gray, d_gray, graySize, cudaMemcpyDeviceToHost));

    // Liberazione memoria
    free(rgb);
    free(gray);
    CHECK(cudaFree(d_rgb));
    CHECK(cudaFree(d_gray));
    */
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <directory_path>\n", argv[0]);
        return 1;
    }

    // Imposta il device CUDA
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev)); // Ottiene le proprietà del dispositivo CUDA
    CHECK(cudaSetDevice(0)); // Seleziona il dispositivo CUDA

    DIR *dir;
    struct dirent *entry;
    if ((dir = opendir(argv[1])) == NULL) {
        perror("opendir");
        return 1;
    }

    // Scansione della cartella
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG) {  // Controlla se è un file regolare
            // Costruisce il percorso completo
            char file_path[1024];
            snprintf(file_path, sizeof(file_path), "%s/%s", argv[1], entry->d_name);
            printf("Processing file: %s\n", file_path);

            // Processa solo i file .fits
            if (strstr(file_path, ".fits") != NULL || strstr(file_path, ".fit")) {
                process_fits_image(file_path, entry->d_name);
            }
        }
    }

    closedir(dir);
    CHECK(cudaDeviceReset());
    return 0;
}
