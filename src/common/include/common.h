// common.h

#ifndef COMMON_H
#define COMMON_H

#include <ctime>
#include <cstring>

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

#endif // COMMON_H