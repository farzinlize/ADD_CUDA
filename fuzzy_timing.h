#ifndef _FUZZY_TIMING_
#define _FUZZY_TIMING_

#include <Windows.h>
#include <stdint.h> // portable: uint64_t   MSVC: __int64 

// typedef struct timeval {
//     long tv_sec;
//     long tv_usec;
// } timeval;

int gettimeofday(struct timeval * tp, struct timezone * tzp);
void set_clock();
double get_elapsed_time();

#endif