#include "kernels.cuh"

#ifdef IN_ARRAY
__global__ void add_kernel_in_array(int *a_in)
{
	extern __shared__ int a_s[];
	unsigned int tid_block = threadIdx.x;
	unsigned int tid = (blockDim.x*2) * blockIdx.x + tid_block;
	
	a_s[tid_block] = a_in[tid] + a_in[tid+blockDim.x];
    __syncthreads();

    for (unsigned int s = blockDim.x/2; s > 0 ; s >>= 1){
		if (tid_block < s)
			a_s[tid_block] = a_s[tid_block] + a_s[tid_block + s];
		__syncthreads();
	}

    if (tid_block == 0)
        a_in[blockIdx.x] = a_s[0];
}

void add_kernel_in_array_wrapper(kernelParameters * params, int * a_in)
{
	add_kernel_in_array<<<params->grid, params->block, params->shared_size, params->stream>>>(a_in);
}
#else
__global__ void add_kernel(int *a_in, int *out)
{
	extern __shared__ int a_s[];
	unsigned int tid_block = threadIdx.x;
	unsigned int tid = (blockDim.x*2) * blockIdx.x + tid_block;
	
	a_s[tid_block] = a_in[tid] + a_in[tid+blockDim.x];
    __syncthreads();

    for (unsigned int s = blockDim.x/2; s > 0 ; s >>= 1){
		if (tid_block < s)
			a_s[tid_block] = a_s[tid_block] + a_s[tid_block + s];
		__syncthreads();
	}

    if (tid_block == 0)
        out[blockIdx.x] = a_s[0];
}

void add_kernel_wrapper(kernelParameters * params, int * a_in, int * out)
{
	add_kernel<<<params->grid, params->block, params->shared_size, params->stream>>>(a_in, out);
}
#endif