#include"helper_functions.h"

void validate(int *a, int *b, int length) {
	for (int i = 0; i < length; ++i) {
		if (a[i] != b[i]) {
			printf("Different value detected at position: %d,"
				" expected %d but get %d\n", i, a[i], b[i]);
			return;
		}
	}
	printf("Tests PASSED successfully! There is no differences\n");
}

void initialize_data_random(int **data, int data_size) {

	static time_t t;
	srand((unsigned)time(&t));

	*data = (int *)malloc(sizeof(int) * data_size);
	for (int i = 0; i < data_size; i++) {
		(*data)[i] = rand() % RANDOM_NUMBER_MAX;
	}
}

void initialize_data_zero(int **data, int data_size) {
	*data = (int *)malloc(sizeof(int) * data_size);
	memset(*data, 0, data_size * sizeof(int));
}

void initialize_data_random_cudaMallocHost(int **data, int data_size){

	static time_t t;
	srand((unsigned) time(&t));

	cudaMallocHost((void **)data, sizeof(int) * data_size);
	for(int i = 0; i < data_size; i++){
		(*data)[i] = rand() % RANDOM_NUMBER_MAX;
	}   
}

void initialize_data_zero_cudaMallocHost(int **data, int data_size){
	cudaMallocHost((void **)data, sizeof(int) * data_size);
	memset(*data, 0, data_size*sizeof(int));
}