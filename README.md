# CUDA Astronomical Image Processing Tools

This project contains two CUDA-accelerated tools for astronomical image processing:
- **CudaStackerAlfaSigma**: Stacks multiple astronomical images using the Alfa-Sigma algorithm
- **CudaStarFinder**: Detects stars in astronomical images

There are also sequential implementation of the used algorithm, used to test and verify results, respectively **CPUStackerAlfaSigma** and **CPUStarFinder**

## CudaStackerAlfaSigma

Stacks multiple FITS images using the Alfa-Sigma algorithm to remove noise, artifacts and improve image quality.
The image files to stack must be in a directory, the name of the directory need to be passed to the executable as parameter.

### Parameters
- `--input-directory, -i` (required): Directory containing FITS files
- `--output-directory, -o` (optional, default .): Output directory path
- `--file-name, -n` (optional, default: image): Output file name
- `--kappa, -k` (optional, default: 3.0): Kappa value for outlier detection
- `--sigma, -s` (optional, default: 5): Number of iterations

### Execution example
```
./cudaStackerAlfaSigma --input-directory ./raw_images --output-directory ./stacked --kappa 2.5
```


## CudaStarFinder

Detects and marks stars in FITS images.

### Parameters

- `--input-file, -f` (required): Path to input FITS file
- `--threshold, -t` (optional, default: 1000): Brightness threshold (0-65535)
- `--reduce-factor, -r` (optional, default: 8): Reduction factor for fast-adaptive mode
- `--threshold-algorithm, -a` (optional, default: simple): Algorithm choice
  - `simple`: Fixed threshold
  - `adaptive`: Adaptive threshold
  - `fast-adaptive`: Fast adaptive threshold
- `--window-size, -w` (optional, default: 255): Window size for adaptive and fast-adaptive threshold
- `--max-star-size, -m` (optional, default: 75): Maximum star diameter in pixels

There will be created 2 fits image in the current directory:
- threshold image
- detected stars

### Example
```bash
./cudaStarFinder --input-file image.fits --threshold 1500 --threshold-algorithm adaptive
```
