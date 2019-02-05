#include "kernels.cuh"

#ifdef IN_ARRAY
__global__ void add_kernel_in_array(arguments args)
#else
__global__ void add_kernel(arguments args)
#endif
{
	extern __shared__ int a_s[];
	unsigned int tid_block = threadIdx.x;
	unsigned int tid = (blockDim.x*2) * blockIdx.x + tid_block;
	
	a_s[tid_block] = args.a_in[tid] + args.a_in[tid+blockDim.x];
    __syncthreads();

    for (unsigned int s = blockDim.x/2; s > 0 ; s >>= 1){
		if (tid_block < s)
			a_s[tid_block] = a_s[tid_block] + a_s[tid_block + s];
		__syncthreads();
	}

	if (tid_block == 0){
		#ifdef IN_ARRAY
		args.a_in[blockIdx.x] = a_s[0];
		#else
		args.out[blockIdx.x] = a_s[0];
		#endif
	}
}