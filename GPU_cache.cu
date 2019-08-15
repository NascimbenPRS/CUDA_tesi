#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>



// sum array of integers sequentially (using cache)
__global__ void arraySum(int *arr, int arraySize, int *sumValue) {
	int tempSum = 0;
		for (int i = 0; i < arraySize; i++) {
			tempSum += arr[i];
		}
	*sumValue = tempSum;
	printf("Sum value (thread): %d\n", *sumValue);
}

// kernel to initialize array
__global__ void initializeArray(int *arr, int arraySize) {
	for (int i = 0; i < arraySize; i++) {
		arr[i] = 1;
	}
	printf("Array initialized\n");
}

int main()
{

	int arraySize = 1 << 22; // 4M integers
	int numCycles = 10; // # of repetitions
	int *arr, *sumValue;

	printf("Sum array of integers on GPU (single thread) using cache.\nArray size=  %d integers\n", arraySize);
	

	cudaMallocManaged(&arr, arraySize * sizeof(int)); // allocate arraySize * 4 bytes
	cudaMallocManaged(&sumValue, sizeof(int));

	// initialize array

	/* initialize on CPU
	for (int i = 0; i < arraySize; i++) {
		arr[i] = 1;
	}

	*/
	

	initializeArray << <1, 1 >> > (arr, arraySize);
	cudaDeviceSynchronize();

	*sumValue = 0;

	/*
	// Prefetch data to GPU
	int device = -1;
	cudaGetDevice(&device); 
	cudaDeviceSynchronize();
	cudaMemPrefetchAsync(sumValue, sizeof(int), device, NULL);
	cudaDeviceSynchronize();
	printf("Device used: %d\n", device);
	*/

	// Time measurement
	double elapsedTime;
	int elapsedClocks = 0, startClock = 0, endClock = 0;
	startClock = clock();

	for (int i= 0; i < numCycles; i++){
		arraySum << <1, 1 >> > (arr, arraySize, sumValue);
		cudaDeviceSynchronize();
		//printf("Call number: %d\n", i + 1);
	}


	endClock = clock();

	// Print results 
	printf("startClock: %d\n", startClock);
	printf("endClock: %d\n", endClock);
	elapsedClocks = endClock - startClock;
	printf("elapsedClock: %d\n", elapsedClocks);
	elapsedTime = ((double)(elapsedClocks)) / (CLOCKS_PER_SEC * numCycles); // average execution time
	printf("Sum = %d, elapsed time= %f s.\n", *sumValue, elapsedTime);


	cudaFree(arr);
	cudaFree(sumValue);
	return 0;
}