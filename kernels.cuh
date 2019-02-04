#ifndef _KERNELS_CUH_
#define _KERNELS_CUH_

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef struct kernelParameters {
    dim3 grid;
    dim3 block;
    int shared_size;
    cudaStream_t stream; 
} kernelParameters;

#ifdef IN_ARRAY
__global__ void add_kernel_in_array(int *a_in);
void add_kernel_in_array_wrapper(kernelParameters * params, int * a_in);
#else
__global__ void add_kernel(int *a_in, int *out);
void add_kernel_wrapper(kernelParameters * params, int * a_in);
#endif

#endif