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

typedef struct KernelAddArguments{
    int size;
    int * a_in;
    #ifndef IN_ARRAY 
    int * out;
    #endif
} arguments;

#ifdef IN_ARRAY
__global__ void add_kernel_in_array(arguments args);
#else
__global__ void add_kernel(arguments args);
#endif

#endif