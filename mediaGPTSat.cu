#include "cuda_runtime.h"
#include <stdio.h>
#include "fitsio.h"
#include <dirent.h>
#include <string.h>

#define CHECK(err) do { cuda_check((err), __FILE__, __LINE__); } while(false)
inline void cuda_check(cudaError_t error_code, const char *file, int line) {
    if (error_code != cudaSuccess) {
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
        fflush(stderr);
        exit(error_code);
    }
}

// accumulo dei valori di ogni pixel delle immagini con SATURAZIONE
__global__ void accumulatePixels(float *d_accum, float *d_image, int npixels, float max_value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < npixels) {
        // Accumulo con saturazione
        float temp = d_accum[idx] + d_image[idx];
        d_accum[idx] = (temp > max_value) ? max_value : temp;
    }
}


// calcolo media finale
__global__ void computeMean(float *d_accum, int numImages, int npixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < npixels) {
        d_accum[idx] /= numImages;
    }
}

void process_fits_images(const char *directory, float **d_accum, int *width, int *height, int *numImages) {
    DIR *dir;
    struct dirent *entry;

    // Open directory
    if ((dir = opendir(directory)) == NULL) {
        perror("opendir");
        exit(1);
    }

    // Initialize image count
    *numImages = 0;

    // Temporary variables for image data
    fitsfile *fptr;
    int status = 0, naxis;
    long naxes[2] = {1, 1};
    size_t npixels;

    float *h_image = NULL;
    float *d_image = NULL;

    // Process all FITS files in directory
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG) {
            char file_path[1024];
            snprintf(file_path, sizeof(file_path), "%s/%s", directory, entry->d_name);

            // Process only .fits files
            if (strstr(file_path, ".fit") != NULL) {
                // Open FITS file
                if (fits_open_file(&fptr, file_path, READONLY, &status)) {
                    fits_report_error(stderr, status);
                    continue;
                }

                // Read image dimensions
                if (fits_get_img_dim(fptr, &naxis, &status) || fits_get_img_size(fptr, 2, naxes, &status)) {
                    fits_report_error(stderr, status);
                    fits_close_file(fptr, &status);
                    continue;
                }

                if (*numImages == 0) {
                    // First image: allocate accum buffer on device
                    *width = naxes[0];
                    *height = naxes[1];
                    npixels = (*width) * (*height);
                    // La memoria allocata ha una dimensione pari al numero di pixel (npixels) moltiplicato per la dimensione di un numero in virgola mobile (sizeof(float))
                    CHECK(cudaMalloc(d_accum, npixels * sizeof(float)));
                    // Impostare tutti i valori nella memoria GPU a zero
                    CHECK(cudaMemset(*d_accum, 0, npixels * sizeof(float)));
                } else if (naxes[0] != *width || naxes[1] != *height) {
                    // Skip images with different dimensions
                    fprintf(stderr, "Skipping file %s due to mismatched dimensions.\n", file_path);
                    fits_close_file(fptr, &status);
                    continue;
                }

                // Allocate host and device buffers for current image
                h_image = (float *)malloc(npixels * sizeof(float));
                CHECK(cudaMalloc(&d_image, npixels * sizeof(float)));

                // Read FITS image data
                if (fits_read_img(fptr, TFLOAT, 1, npixels, NULL, h_image, NULL, &status)) {
                    fits_report_error(stderr, status);
                    fits_close_file(fptr, &status);
                    free(h_image);
                    CHECK(cudaFree(d_image));
                    continue;
                }
                fits_close_file(fptr, &status);

                // Copy image to device
                CHECK(cudaMemcpy(d_image, h_image, npixels * sizeof(float), cudaMemcpyHostToDevice));

                // Accumulate image on GPU
                float max_pixel_value = 1.0e5; // Soglia massima, definita in base al tuo contesto (MODIFICARE!!!!!!!!!!!!!)
                int blockSize = 256;
                int gridSize = (npixels + blockSize - 1) / blockSize;
                accumulatePixels<<<gridSize, blockSize>>>(*d_accum, d_image, npixels, max_pixel_value);
                CHECK(cudaDeviceSynchronize());

                // Clean up
                free(h_image);
                CHECK(cudaFree(d_image));

                (*numImages)++;
            }
        }
    }
    closedir(dir);
}

void save_fits_image(const char *output_path, float *d_accum, int width, int height) {
    fitsfile *fptr;
    int status = 0;
    long naxes[2] = {width, height};
    size_t npixels = width * height;

    // Allocate host buffer for final image
    float *h_final = (float *)malloc(npixels * sizeof(float));
    CHECK(cudaMemcpy(h_final, d_accum, npixels * sizeof(float), cudaMemcpyDeviceToHost));

    // Create output FITS file
    if (fits_create_file(&fptr, output_path, &status)) {
        fits_report_error(stderr, status);
        free(h_final);
        return;
    }

    if (fits_create_img(fptr, FLOAT_IMG, 2, naxes, &status)) {
        fits_report_error(stderr, status);
        free(h_final);
        fits_close_file(fptr, &status);
        return;
    }

    // Write image data
    if (fits_write_img(fptr, TFLOAT, 1, npixels, h_final, &status)) {
        fits_report_error(stderr, status);
    }

    // Clean up
    fits_close_file(fptr, &status);
    free(h_final);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Usage: %s <input_directory> <output_file>\n", argv[0]);
        return 1;
    }

    const char *input_dir = argv[1];
    const char *output_file = argv[2];

    int width, height, numImages;
    float *d_accum = NULL;

    // Process images and compute average
    process_fits_images(input_dir, &d_accum, &width, &height, &numImages);

    if (numImages > 0) {
        // Compute final mean on GPU
        size_t npixels = width * height;
        int blockSize = 256;
        int gridSize = (npixels + blockSize - 1) / blockSize;
        computeMean<<<gridSize, blockSize>>>(d_accum, numImages, npixels);
        CHECK(cudaDeviceSynchronize());

        // Save final image
        save_fits_image(output_file, d_accum, width, height);
    } else {
        printf("No valid FITS images found.\n");
    }

    // Clean up
    CHECK(cudaFree(d_accum));
    CHECK(cudaDeviceReset());
    return 0;
}

/*
/// 1 Kernel CUDA
a) accumulatePixels:
    - Somma i valori di un'immagine (d_image) a un buffer di accumulo (d_accum).
    - Gli indici dei pixel sono calcolati usando i parametri del kernel (threadIdx, blockIdx, blockDim).
    - Esegue solo su pixel validi (indice < npixels).
b) computeMean:
    - Calcola la media dei pixel nel buffer di accumulo d_accum, dividendo ciascun pixel per il numero totale di immagini.

/// 2 process_fits_images()
a) Questa funzione elabora tutte le immagini FITS in una directory.
b) Input:
    - directory: Path della directory con i file FITS.
    - d_accum: Puntatore a buffer GPU per il risultato.
    - width, height: Dimensioni delle immagini (devono essere uniformi).
    - numImages: Conteggio delle immagini valide.
c) Funzionamento:
    - Apre la directory specificata.
    - Itera su tutti i file e considera solo quelli con estensione .fits.
    - Per ogni file FITS:
        Usa la libreria CFITSIO per leggere le dimensioni e i dati dell'immagine.
        Alloca memoria su host (CPU) e device (GPU) per i pixel.
        Copia l'immagine su GPU e usa il kernel accumulatePixels per aggiungere i pixel al buffer di accumulo.
        Libera le risorse temporanee.
    - Ignora immagini con dimensioni diverse dalla prima trovata.

/// 3 save_fits_image()
a) Salva il risultato finale come file FITS.
b) Input:
    - output_path: Path del file di output.
    - d_accum: Buffer GPU con i dati da salvare.
    - width, height: Dimensioni dell'immagine.
c) Funzionamento:
    - Copia i dati da GPU a CPU.
    - Usa CFITSIO per creare un nuovo file FITS e scrivere i dati.

/// 4 main()
a) Verifica che l'utente abbia fornito due argomenti: una directory di input e un file di output.
b) Chiama process_fits_images() per elaborare le immagini.
c) Se sono state trovate immagini valide:
    - Usa computeMean per calcolare la media pixel-per-pixel.
    - Salva il risultato con save_fits_image().
d) Libera le risorse GPU.
*/