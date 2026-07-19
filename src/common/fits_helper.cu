#include "fits_helper.h"
#include "cuda_helper.h"

#include <dirent.h>
#include <string.h>

int load_images_to_memory_prefetch(const char *dir_path, u_int16_t *img_all,
                                   long width, long height, long n_chan,
                                   int count, int dev) {
    DIR *dir = opendir(dir_path);
    if (!dir) {
        perror("opendir");
        return 1;
    }

    PrefetchDeviceArg devLoc = make_prefetch_device_arg(dev);

    struct dirent *entry;
    int status = 0, idx = 0;
    long w, h, n;
    long data_size = width * height * n_chan;
    while ((entry = readdir(dir)) != NULL && idx < count) {
        if (entry->d_type != DT_REG)
            continue;
        if (!(strstr(entry->d_name, ".fits") || strstr(entry->d_name, ".fit")))
            continue;

        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);

        fitsfile *fptr = nullptr;
        open_fits(path, &fptr);

        get_fits_dimensions(fptr, &w, &h, &n);
        if (w != width || h != height || n != n_chan)
            continue;

        get_fits_data(fptr, data_size, img_all + idx * data_size);
        fits_close_file(fptr, &status);

        CHECK(cudaMemPrefetchAsync(img_all + idx * data_size,
                                   data_size * sizeof(u_int16_t),
                                   devLoc, 0));
        idx++;
    }
    if (idx != count)
        printf("  Warning! Number of expected images: %d, Actually loaded: %d\n", count, idx);
    closedir(dir);
    return 0;
}