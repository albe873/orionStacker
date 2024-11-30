
README: CUDA Stacker Alfa Sigma

Descrizione del Programma
-------------------------
Il codice processa una serie di immagini FITS contenute in una directory, accumula i valori pixel per pixel, 
calcola la media e può anche calcolare la deviazione standard delle immagini. Utilizza l'accelerazione GPU tramite CUDA.

Il flusso del programma è suddiviso in più fasi come descritto di seguito:

1. Inclusione delle librerie
   - Librerie necessarie:
     - cuda_runtime.h: Per gestire le operazioni CUDA.
     - fitsio.h: Per leggere e scrivere file FITS.
     - stb_image.h / stb_image_write.h: Per manipolare immagini.
   - Viene definita una macro `CHECK` per verificare errori CUDA e gestirli in modo sicuro.

2. Funzioni principali
   a. **rgbToGrayCPU / rgbToGrayGPU**:
      - Convertitori da immagine RGB a scala di grigi per CPU e GPU.

   b. **accumulatePixels (CPU e GPU)**:
      - Accumula i valori di più immagini pixel per pixel.
      - La versione GPU utilizza una combinazione di blocchi e thread per parallelizzare il calcolo.

   c. **computeMean / computeMeanAdv**:
      - Calcolano la media pixel per pixel. 
      - `computeMeanAdv` supporta immagini con valori zero, ignorandoli dal calcolo.

   d. **computeStdDev**:
      - Calcola la deviazione standard per ogni pixel considerando più immagini.

   e. **filterPixels**:
      - rimuove pixel che si trovano al di fuori di un intervallo accettabile definito dalla loro media e deviazione standard.
      - Per ogni immagine:
         Controlla se il valore del pixel corrente (image[i][idx]) è fuori dall'intervallo definito da:
            // Intervallo accettabile = [mean[idx] − k⋅std[idx], mean[idx] + k⋅std[idx]]
         Se il valore del pixel non rientra nell'intervallo, viene impostato a 0 (filtrato).

3. Funzioni di supporto per immagini FITS
   a. **open_fits**: Apre un file FITS.
   b. **get_image_dimensions**: Ottiene larghezza, altezza e profondità di un'immagine FITS.
   c. **get_fits_data**: Legge i dati dell'immagine FITS.
   d. **save_image_fits**: Salva un'immagine in formato FITS.

4. Scansione della directory
   - Viene scansionata una directory specificata come argomento per identificare file con estensione `.fits`.
   - Le immagini FITS vengono caricate e validate per garantire che abbiano le stesse dimensioni.

5. Allocazione memoria GPU
   - Usa `cudaMallocManaged` per allocare memoria unificata per i dati delle immagini, la media e la deviazione standard.
   - cudaMemAdvise: Suggerisce al sistema di preferire la GPU (dev) come posizione preferita per accumulatore, media e deviazione standard.

6. Calcolo GPU
   - Calcola la media usando `computeMeanAdv` su GPU.
   - La deviazione standard è commentata nel codice, ma può essere calcolata con `computeStdDev`.

7. Salvataggio del risultato
   - Salva la media calcolata in un nuovo file FITS usando `save_image_fits`.

8. Pulizia
   - Libera la memoria GPU con `cudaFree` e reimposta il dispositivo CUDA.

Requisiti
----------
1. NVIDIA CUDA Toolkit installato.
2. CFITSIO Library per operazioni sui file FITS.
3. Un compilatore C compatibile con CUDA (ad esempio, `nvcc`).
4. Immagini FITS nella directory di input.

Istruzioni per la compilazione
------------------------------
1. Per compilare il codice con CUDA, usa:
   ```bash
   nvcc cudaStackerAlfaSigma.cu -o cudaStackerAlfaSigma -lcfitsio -O3 -Wall
   ```
2. Se si utilizza AMD HIP, usa HIPify per convertire il codice e compilarlo:
   ```bash
   hipify-clang cudaStackerAlfaSigma.cu --cuda-path=/opt/cuda
   hipcc cudaStackerAlfaSigma.cu.hip -o cudaStackerAlfaSigma -lcfitsio -O3 -Wall
   ```

Istruzioni per l'esecuzione
---------------------------
1. Esegui il programma specificando la directory contenente le immagini FITS:
   ```bash
   ./cudaStackerAlfaSigma <path_to_directory>
   ```
2. Il risultato verrà salvato in `output/output.fits`.

Note aggiuntive
---------------
- Assicurarsi che tutte le immagini FITS nella directory abbiano le stesse dimensioni.
- Il programma include error handling per immagini non valide o di dimensioni non corrispondenti.

Contatti
--------
Per problemi o domande, contattare il manutentore del codice.
