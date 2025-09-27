# ORION - Optical Reconstruction & Image Operation for Night-Sky

This project contains two CUDA-accelerated tools for astronomical image processing:
- **CudaStackerAlfaSigma**: Stacks multiple astronomical images using the Alfa-Sigma algorithm
- **CudaStarFinder**: Detects stars in astronomical images

There are also sequential implementation of the used algorithm, used to test and verify results, respectively **CPUStackerAlfaSigma** and **CPUStarFinder**

## CudaStackerAlfaSigma

Stacks multiple FITS images using the Alfa-Sigma algorithm to improve image quality.
This method is used in image stacking to suppress outliers (such as noise, cosmic rays, hot pixels, or transient artifacts) and compute a clean, representative average of multiple exposures of the same scene. It is particularly useful in astrophotography and astronomical imaging, where datasets often include several images of the same target, but with varying noise and artifacts.

In our CUDA version, the algorithm is designed to process two pixels per thread, achieving fine-grained parallelism across the entire image. By combining iterative statistics and conditional filtering in each thread, the implementation performs complex per-pixel operations efficiently on the GPU, making it ideal for large-scale image stacking tasks.
For each pixel pair, the following steps are executed:
- **Initial Mean Estimation:** A partial mean is computed across the pixel values from all input images, excluding zero values, which typically represent masked or filtered pixels.
- **Standard Deviation Computation:** The standard deviation of the non-zero pixel values is calculated to quantify how much each pixel deviates from the mean.
- **Outlier Rejection (Kappa Clipping):** Each pixel value is compared against a dynamic threshold determined by the formula:

    Threshold = μ ± k ⋅ σ

    where:
    - μ is the mean
    - σ is the standard deviation
    - k is the kappa factor (user-defined, e.g., 3.0)

Pixel values falling outside this range are considered outliers and are set to zero (excluded in future iterations).
- **Iterative Refinement:** The algorithm repeats the partial mean and standard deviation computation, followed by outlier filtering, for a fixed number of iterations (sigma). This process gradually refines the pixel set, progressively removing outliers.
- **Final Mean Computation:** After all iterations, a final mean is computed over the remaining valid (non-zero) pixel values, yielding the output pixel value for the stacked image.

Unlike a simple arithmetic mean, Alfa Sigma effectively removes sporadic high or low pixel values that would otherwise bias the result. The iterative clipping ensures that only statistically inconsistent data is excluded, preserving valid signal.
The user-defined kappa (clipping strength) and sigma (number of iterations) allow tuning the aggressiveness of the filtering, depending on the noise characteristics of the dataset.

The image files to stack must be in a directory, the name of the directory need to be passed to the executable as parameter.

The first image file of the directory will be used to determine the dimensions accepted by the stacker.

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

Detects and marks stars in FITS images. It converts RGB images to grayscale directly on the GPU for efficient preprocessing and implements three thresholding methods.

The input FITS images are in a planar format where color channels (red, green, blue) are stored separately. The to_grayscale_fits kernel converts these RGB images to grayscale by calculating the luminance using standard weighted coefficients.

Two thresholding approaches are implemented for segmenting star candidates:
- **Simple Thresholding:** The simple_threshold kernel compares each pixel’s intensity against a fixed threshold, setting pixels above the threshold to their original value and others to zero.
- **Adaptive Thresholding:** The adaptiveThresholdingKernel performs a local, window-based thresholding where the threshold is dynamically computed as the mean intensity of a surrounding window plus an offset. This allows for varying background illumination and more robust star detection in uneven images.

To further improve performance, an **approximate adaptive thresholding** method is provided, which downsamples the image using the reduce_image kernel. This smaller “reduced” image is used to calculate local means faster in the adaptiveThresholdingApprossimative kernel, enabling scalable processing of very large images.

The star detection tracks progress independently along four directions (right, down, left, up) during the spiral traversal. For each non-black pixel, it checks if it is a local maximum of brightness.

The kernel performs a spiral walk around the pixel, moving right, down, left, and up, expanding the search area step-by-step. In each direction, it verifies that pixels are black (intensity zero), confirming the star is isolated by a black background. It dynamically adapts step limits and halts exploration in directions fully surrounded by black pixels, reducing unnecessary computations.
To avoid detecting the same star multiple times, if another pixel with the same brightness but a higher index is found, the current pixel is rejected.

The final detected star is outlined by a bounding square centered on the original pixel, with careful boundary checks to avoid out-of-range accesses.

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
