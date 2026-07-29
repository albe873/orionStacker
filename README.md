# ORIONStacker

An astrophotography stacking program that leverages parallel GPU computing through CUDA C to accelerate processing.

The program was developed to perform stacking of FITS images using the Alpha-Sigma algorithm, with the goal of improving the quality of the final image by enhancing the signal of the photographed object and reducing the noise present in the individual shots. Thanks to these features, it is particularly useful in astrophotography and astronomical image processing, where numerous acquisitions of the same subject affected by different levels of noise and disturbance are available.

## Project structure

```
orionStacker/
├── .devcontainer/     # Configuration for container-based development (Dockerfile)
├── src/               # Main source code
│   ├── calibration/   # Image calibration (host/CPU and device/CUDA versions)
│   ├── common/        # Shared utilities: FITS file handling, CUDA helpers, stb_image libraries
│   ├── debayer/       # Image debayering (MHC filters)
│   ├── gui/           # Application graphical interface
│   ├── stacker/       # Implementation of the Alpha-Sigma stacking algorithm
│   ├── star_finder/   # Star detection (thresholding, Otsu, warp, descriptors)
│   └── utils/         # Small command-line tools (e.g. FITS metadata reading, result verification)
├── test/              # Tests and benchmarks for the various modules (stacker, star finder, calibration, debayer, aligner)
├── third_party/       # External dependencies (e.g. ADE, OpenCV) included as submodules, with build scripts
├── CMakeLists.txt     # Main configuration for building with CMake
└── README.md
```

The project mainly handles calibration of frames obtained directly from the astro camera, debayering of the calibrated light frames, alignment through star detection in the images, and finally stacking of the final result.

## Data

- **Input format:** 16-bit unsigned FIT images.
- **Output format:** 16-bit unsigned FIT images.

**Required frame types:**

- **Bias frames:** used to compute the master bias (average of the bias frames)
- **Dark frames:** used to compute the master dark
- **Flat frames:** used, together with the master bias, to compute the master flat
- **Light frames:** corrected using the master bias, master dark, and master flat

## Building

**Required dependencies:** the project uses CMake + OpenCV + cfitsio

```bash
sudo apt update
sudo apt install -y build-essential cmake libcfitsio-dev libopencv-dev
```

**Build with CMake**

```bash
git clone --branch dev https://github.com/albe873/orionStacker.git
cd orionStacker
mkdir build && cd build
cmake ..
cmake --build . -j$(nproc)
```

**Running the various components**

```bash
cd orionStacker/build/test
```

*CudaCalibration*

```bash
./cudaCalibration --light /path/light/ --bias /path/bias/ --dark /path/dark/ --flat /path/flat/ --output /path/output/
./cudaCalibration --base-dir /path/ --output /path/output/
```

*CudaDebayering*

```bash
./cudaDebayerTest --input /path/raw_bayer/ --output /path/output/
```

*CudaAlligner*

```bash
./cudaAligner --input-file1 /path/img1.fits --input-file2 /path/img2.fits --descriptor-neighbors 2
```

*CudaStackerAlfaSigma*

```bash
./cudaStackerAlfaSigma --input-directory /path/calibrated_lights/ --output-directory /path/output/ --file-name stack --kappa 3.0 --iterations 5
```

*CudaStarFinder*

```bash
./cudaStarFinder --input-file /path/image.fits --threshold-algorithm adaptive --window-size 201 --max-star-size 100 --min-star-size 4
```

**Full run**

```bash
./testAll --light /path/light/ --bias /path/bias/ --dark /path/dark/ --flat /path/flat/ --output /path/output/
./testAll --base-dir /path/ --output /path/output/
```

> **Work in progress:** the pipeline is already active, the GUI is catching up.

## Pipeline workflow

1. Input calibration frames and light frames
2. Calibration of the light frames
3. Debayering
4. Alignment of calibrated light frames
5. Stacking of calibrated light frames

![Pipeline](/pipeline.jpg)

## Screenshot / Risultati