#ifndef __HELPER_FUNCTIONS_
#define __HELPER_FUNCTIONS_

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include<cuda.h>

#define RANDOM_NUMBER_MAX 5

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

void validate(int *a, int *b, int length);
void initialize_data_random(int **data, int data_size);
void initialize_data_zero(int **data, int data_size);
void initialize_data_random_cudaMallocHost(int **data, int data_size);
void initialize_data_zero_cudaMallocHost(int **data, int data_size);

#endif