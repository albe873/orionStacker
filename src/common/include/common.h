// common.h

#ifndef COMMON_H
#define COMMON_H

#include <ctime>
#include <cstring>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

inline double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

inline void remove_trailing_slash(char *in_dir) {
    if (in_dir[strlen(in_dir) - 1] == '/') {
        in_dir[strlen(in_dir) - 1] = '\0';
    }
}

inline int is_regular_file(const char *path) {
    struct stat path_stat;
    stat(path, &path_stat);
    return S_ISREG(path_stat.st_mode);
}

inline int is_directory(const char *path) {
    struct stat path_stat;
    stat(path, &path_stat);
    return S_ISDIR(path_stat.st_mode);
}

#endif // COMMON_H