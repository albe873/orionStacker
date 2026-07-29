# ORIONStacker

Programma di stacking astrofotografico che sfrutta il calcolo parallelo delle GPU tramite CUDA C per accelerare l'elaborazione.

Il programma è stato sviluppato per eseguire lo stacking di immagini FITS mediante l'algoritmo Alfa-Sigma, con l'obiettivo di migliorare la qualità dell'immagine finale, mettendo in evidenza il segnale dell'oggetto fotografato e riducendo il rumore presente nei singoli scatti. Per queste caratteristiche, è particolarmente utile nell'astrofotografia e nell'elaborazione di immagini astronomiche, dove sono disponibili numerose acquisizioni dello stesso soggetto affette da differenti livelli di rumore e disturbi.

## Struttura del progetto

```
orionStacker/
├── .devcontainer/     # Configurazione per lo sviluppo in container (Dockerfile)
├── src/               # Codice sorgente principale
│   ├── calibration/   # Calibrazione delle immagini (versioni host/CPU e device/CUDA)
│   ├── common/        # Utilità condivise: gestione file FITS, helper CUDA, librerie stb_image
│   ├── debayer/       # Debayering delle immagini (filtri MHC)
│   ├── gui/           # Interfaccia grafica dell'applicazione
│   ├── stacker/       # Implementazione dell'algoritmo di stacking Alfa-Sigma
│   ├── star_finder/   # Rilevamento delle stelle (thresholding, Otsu, warp, descrittori)
│   └── utils/         # Piccoli tool da linea di comando (es. lettura metadati FITS, verifica risultati)
├── test/              # Test e benchmark dei vari moduli (stacker, star finder, calibrazione, debayer, aligner)
├── third_party/       # Dipendenze esterne (es. ADE, OpenCV) incluse come submodule, con script di build
├── CMakeLists.txt     # Configurazione principale per la build con CMake
└── README.md
```

Il progetto si occupa principalmente di calibrazione dei frame ottenuti direttamente dall'astro camera, debayering dei light calibrati, allineamento tramite rilevamento delle stelle nelle immagini ed infine stacking del risultato finale.

## Dati

- **Formato input:** immagini FIT a 16 bit unsigned.
- **Formato output:** immagini FIT a 16 bit unsigned.

**Tipi di frame richiesti:**

- **Bias frames:** usati per calcolare il master bias (media dei bias)
- **Dark frames:** usati per calcolare il master dark
- **Flat frames:** usati, insieme al master bias, per calcolare il master flat
- **Light frames:** vengono corrette usando master bias, master dark e master flat

## Compilazione

**Dipendenze necessarie:** il progetto usa CMake + OpenCV + cfitsio

```bash
sudo apt update
sudo apt install -y build-essential cmake libcfitsio-dev libopencv-dev
```

**Build con CMake**

```bash
git clone --branch dev https://github.com/albe873/orionStacker.git
cd orionStacker
mkdir build && cd build
cmake ..
cmake --build . -j$(nproc)
```

**Esecuzione dei vari componenti**

```bash
cd orionStacker/build/test
```

*CudaCalibration*

```bash
./cudaCalibration --light /percorso/light/ --bias /percorso/bias/ --dark /percorso/dark/ --flat /percorso/flat/ --output /percorso/output/
./cudaCalibration --base-dir /percorso/ --output /percorso/output/
```

*CudaDebayering*

```bash
./cudaDebayerTest --input /percorso/raw_bayer/ --output /percorso/output/
```

*CudaAlligner*

```bash
./cudaAligner --input-file1 /percorso/img1.fits --input-file2 /percorso/img2.fits --descriptor-neighbors 2
```

*CudaStackerAlfaSigma*

```bash
./cudaStackerAlfaSigma --input-directory /percorso/calibrated_lights/ --output-directory /percorso/output/ --file-name stack --kappa 3.0 --iterations 5
```

*CudaStarFinder*

```bash
./cudaStarFinder --input-file /percorso/immagine.fits --threshold-algorithm adaptive --window-size 201 --max-star-size 100 --min-star-size 4
```

**Esecuzione completa**

```bash
./testAll --light /percorso/light/ --bias /percorso/bias/ --dark /percorso/dark/ --flat /percorso/flat/ --output /percorso/output/
./testAll --base-dir /percorso/ --output /percorso/output/
```

> **Work in progress:** la pipeline è già attiva, la GUI la sta raggiungendo.

## Pipeline funzionamento

1. Input calibration frames e light frames
2. Calibrazione dei light
3. Debayering
4. Allineamento light calibrati
5. Stacking light calibrati

![Pipeline](/pipeline.jpg)

## Screenshot / Risultati