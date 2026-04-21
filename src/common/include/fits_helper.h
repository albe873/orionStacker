#ifndef FITS_HELPER_H
#define FITS_HELPER_H

#include <fitsio.h>
#include <string>
#include "opencv2/imgcodecs.hpp"


using namespace std;

/*
 * funzioni per facilitare operazioni di lettura/scrittura dei file fits
 * assumo immagini a 16 bit unsigned (ushort)
 * le immagini possono avere 1 o 3 canali (1 canale per grayscale o per raw, 3 canali per RGB)
 *
 * la libreria fitsio dovrebbe gestire conversioni di tipo automaticamente,
 * quindi dovrebbe essere possibile aprire file con tipi diversi (es. byte, short, float)
 * --- non testato ---
 */


/* apre un file fits
*/ 
void open_fits(string file_path, fitsfile **fptr);


/* ottiene le dimensioni dell'immagine fits
*/ 
void get_fits_dimensions(fitsfile *fptr, long *width, long *height, long *n_chan);


/* legge i dati dell'immagine fits
 * *** IMPORTANTE: fits_data deve essere allocato prima della chiamata ***
*/ 
void get_fits_data(fitsfile *fptr, size_t npixels, u_int16_t *fits_data);

/* stampa i metadati del file fits
*/
void print_fits_metadata(fitsfile *fptr);

/* salva un'immagine in formato fits
 * Nota: l'immagine deve essere in formato planare
*/
void save_image_fits(string output_dir_path, string file_name, u_int16_t *image_data, long width, long height, long n_chan);

/* salva un'immagine in formato tiff, 
 * per maggiore compatibilità in fasi future di editing/visualizzazione
 * IMPORTANTE: assumo di aver letto prima un file fits, quindi i dati devono essere forniti
 * in formato planare
 * uso di OpenCV per la scrittura dei dati raw
*/
void save_image_tiff(string output_dir_path, string file_name, u_int16_t *image_data, long width, long height, long n_chan);
#endif // FITS_HELPER_H
