#include <stdio.h>
#include "kernels.cuh"
extern "C"{
    #include "fuzzy_timing.h"
    #include "helper_functions.h"
}

int sum_array(int *a_in, int size)
{
    int sum = a_in[0];
    for(int i = 1 ; i < size ; i++)
        sum += a_in[i];
    return sum;
}

#if defined(OVERLAP)
int overlaped_transfer_kernel(int factor, int stream_count)
{
    /* inform in array or out array operation */
    #if defined(IN_ARRAY)
    printf("[ARRAY] in array operation\n");
    #else
    printf("[ARRAY] out array operation\n");
    #endif

    /* define and set variables */
	int *a_h, *device_out_h;
	int sum_parralel, sum_seq;
    double seq_time, total_time, kernel_time;

    int size = 1024 * 1024 * factor;
    int block_size = 1024;
    int stream_size = size / stream_count;
    int block_count = (stream_size/block_size)/2;

    /* define and set kernel variables */
	dim3 grid_dim(block_count, 1, 1);
	dim3 block_dim(block_size, 1, 1);

    arguments* args = (arguments *)malloc(sizeof(arguments) * stream_count);
    cudaStream_t* streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * stream_count);
	for(int i=0;i<stream_count;i++){
        cudaStreamCreate(&streams[i]);

        /* inital data on device for each stream */
        CUDA_CHECK_RETURN(cudaMalloc((void **)&args[i], sizeof(arguments)));
        CUDA_CHECK_RETURN(cudaMalloc((void **)&(args[i].a_in), sizeof(int)*stream_size));

        #ifndef IN_ARRAY
        CUDA_CHECK_RETURN(cudaMalloc((void **)&(args[i].out), sizeof(int) * block_count));    
        #endif
    }

    /* inital data on host */
    initialize_data_random_cudaMallocHost(&a_h, size);
    initialize_data_zero_cudaMallocHost(&device_out_h, block_count * stream_count);

    /* ### SEQUENTIAL ### */
    set_clock();
	sum_seq = sum_array(a_h, size);
    seq_time = get_elapsed_time();
        
    /* ### PARALLEL GPU ### */
	set_clock();

	int offset = 0, out_offset = 0;
	for(int stream_id=0 ; stream_id < stream_count ; stream_id++){
        cudaMemcpyAsync(args[stream_id].a_in, &a_h[offset], stream_size*sizeof(int), cudaMemcpyHostToDevice, streams[stream_id]);

        #ifdef IN_ARRAY
        add_kernel_in_array<<<grid_dim, block_dim, block_size*sizeof(int), streams[stream_id]>>>(args[stream_id]);
		cudaMemcpyAsync(&device_out_h[out_offset], args[stream_id].a_in, block_count*sizeof(int), cudaMemcpyDeviceToHost, streams[stream_id]);
        #else
		add_kernel<<<grid_dim, block_dim, block_size*sizeof(int), streams[stream_id]>>>(args[stream_id]);
		cudaMemcpyAsync(&device_out_h[out_offset], args[stream_id].out, block_count*sizeof(int), cudaMemcpyDeviceToHost, streams[stream_id]);
        #endif

        offset+=stream_size;
        out_offset+=block_count;
    }
    
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaGetLastError());

    kernel_time = get_elapsed_time();
    set_clock();

	sum_parralel = sum_array(device_out_h, block_count * stream_count);

	total_time = get_elapsed_time();
	total_time += kernel_time;

    /* printing result and validation */
    printf("[TIME] Sequential: %.4f\n", seq_time);
	printf("[TIME] total parallel: %.4f\n", total_time);
    printf("[TIME] kernel_time : %.4f\n", kernel_time);
    printf("[SPEEDUP] sequentianal / parallel_time: %.4f\n", seq_time/total_time);
    printf("[VALIDATE] Parallel_sum: %d \tSeq_sum: %d\n", sum_parralel, sum_seq);
    printf("[VALIDATE] diffrentc of sums: %d\n", abs(sum_parralel - sum_seq));

    /* free alocated memory */
    for(int i=0;i<stream_count;i++){
        cudaStreamDestroy(streams[i]);

        /* inital data on device for each stream */
        CUDA_CHECK_RETURN(cudaFree(args[i].a_in));
        
        #ifndef IN_ARRAY
        CUDA_CHECK_RETURN(cudaFree(args[i].out));
        #endif  
    }

    free(streams);
    free(args);
    CUDA_CHECK_RETURN(cudaFreeHost(a_h));
    CUDA_CHECK_RETURN(cudaFreeHost(device_out_h));

    return 0;
}

int main(int argc, char * argv[])
{
    printf("[MAIN] OVERLAP MAIN\n");

    /* check and warning for user input */
    if(argc != 3){
		printf("Correct way to execute this program is:\n");
		printf("add_cuda factor(MB) stream_count\n");
		printf("For example:\nadd_cuda 40 4\n");
		return 1;
	}

    int factor = atoi(argv[1]);
    int stream_count = atoi(argv[2]);

    return overlaped_transfer_kernel(factor, stream_count);
}

#elif defined(TEST)
int main()
{
    printf("[MAIN] TEST MAIN\n");

    set_clock();

    return 0;
}

#else
int one_add_kernel(int factor)
{
    /* inform in array or out array operation */
    #if defined(IN_ARRAY)
    printf("[ARRAY] in array operation\n");
    #else
    printf("[ARRAY] out array operation\n");
    #endif

    /* define and set variables */
	int *a_h, *device_out_h;
	int sum_parralel, sum_seq;
    double seq_time, total_time, kernel_time, mem_time;

    int size = 1024 * 1024 * factor;
    int block_size = 1024;
    int block_count = (size/block_size)/2;

    /* define and set kernel variables */
	dim3 grid_dim(block_count, 1, 1);
	dim3 block_dim(block_size, 1, 1);

    /* inital data on host */
    initialize_data_random(&a_h, size);
    initialize_data_zero(&device_out_h, block_count);
    
    /* inital data on device */
    arguments arg;
    CUDA_CHECK_RETURN(cudaMalloc((void **)&arg, sizeof(arguments)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&(arg.a_in), sizeof(int) * size));

    #ifndef IN_ARRAY
    CUDA_CHECK_RETURN(cudaMalloc((void **)&(arg.out), sizeof(int) * block_count));    
    #endif

    /* ### SEQUENTIAL ### */
    set_clock();
	sum_seq = sum_array(a_h, size);
    seq_time = get_elapsed_time();
        
    /* ### PARALLEL GPU ### */
	set_clock();

    cudaMemcpy(arg.a_in, a_h, size*sizeof(int), cudaMemcpyHostToDevice);

    mem_time = get_elapsed_time();
    set_clock();

    #ifdef IN_ARRAY
    add_kernel_in_array<<<grid_dim, block_dim, block_size*sizeof(int)>>>(arg);
    #else
    add_kernel<<<grid_dim, block_dim, block_size*sizeof(int)>>>(arg);
    #endif

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaGetLastError());

    kernel_time = get_elapsed_time();
    set_clock();

    #ifdef IN_ARRAY
    cudaMemcpy(device_out_h, arg.a_in, block_count*sizeof(int), cudaMemcpyDeviceToHost);
    #else
    cudaMemcpy(device_out_h, arg.out, block_count*sizeof(int), cudaMemcpyDeviceToHost);
    #endif
    
    mem_time += get_elapsed_time();

    set_clock();

	sum_parralel = sum_array(device_out_h, block_count);

	kernel_time += get_elapsed_time();
	total_time = kernel_time + mem_time;

    /* printing result and validation */
    printf("[TIME] Sequential: %.4f\n", seq_time);
	printf("[TIME] total parallel: %.4f\n", total_time);
    printf("[TIME] kernel_time : %.4f\n", kernel_time);
    printf("[TIME] mem_time : %.4f\n", mem_time);
    printf("[SPEEDUP] sequentianal / parallel_time (total time): %.4f\n", seq_time/total_time);
    printf("[SPEEDUP] sequentianal / parallel_time (only operation): %.4f\n", seq_time/kernel_time);
    printf("[VALIDATE] Parallel_sum: %d \tSeq_sum: %d\n", sum_parralel, sum_seq);
    printf("[VALIDATE] diffrentc of sums: %d\n", abs(sum_parralel - sum_seq));

    /* free alocated memory */
    free(a_h);
    free(device_out_h);

    CUDA_CHECK_RETURN(cudaFree(arg.a_in));

    #ifndef IN_ARRAY
    CUDA_CHECK_RETURN(cudaFree(arg.out));
    #endif

    return 0;
}

int main(int argc, char * argv[])
{
    printf("[MAIN] else MAIN (not overlap transfer)\n");
    
    /* check and warning for user input */
    if(argc != 2){
		printf("Correct way to execute this program is:\n");
		printf("add_cuda factor(MB)\n");
		printf("For example:\nadd_cuda 40\n");
		return 1;
	}

    int factor = atoi(argv[1]);

    return one_add_kernel(factor);
}
#endif