#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>



// sum array of integers sequentially (using cache)
__global__ void arraySumGPU(int *arr, int arraySize, int *sumValue, int numCycles) {
	int tempSum = 0;
	for (int k = 0; k < numCycles; k++) {
		tempSum = 0;
		for (int i = 0; i < arraySize; i++) {
			tempSum += arr[i];
		}
	}
	*sumValue = tempSum;
	//printf("Sum value (thread): %d\n", *sumValue);
}

// sum array of integers non-sequentially (not using cache)
__global__ void arraySumStrideGPU(int *arr, int arraySize, int *sumValue, int numCycles, int cacheLineSize) {
	int tempSum = 0;
	for (int k = 0; k < numCycles; k++) {
		tempSum = 0;
		for (int i = 0; i < cacheLineSize; i++) {
			for (int j = i; j < arraySize; j += cacheLineSize) {
				tempSum += arr[j];
			}
		}
	}
	*sumValue = tempSum;
}

// kernel to initialize array
__global__ void initializeArray(int *arr, int arraySize) {
	for (int i = 0; i < arraySize; i++) {
		arr[i] = 1;
	}
	printf("Array initialized\n");
}

int main(int argc, char *argv[])
{
	int arraySize = 1 << 22; // 4M integers
	int usesCache = 1; // 0: don't use cache, 1: use cache (default)
	int cacheLineSize = 128 / sizeof(int); // # integers per cache line
	int numCycles = 5; // # of repetitions
	int *arr, *sumValue;

	cudaMallocManaged(&arr, arraySize * sizeof(int)); // allocate arraySize * 4 bytes
	cudaMallocManaged(&sumValue, sizeof(int));
	printf("Sum array of integers on GPU (single thread). Array size=  %d integers\n", arraySize);


	// check for command line options
	if (argc < 2) {
		printf("No options specified, use cache by default. no_cache option available\n");
	}
	else {
		// options specified
		if (strcmp(argv[1], "no_cache") == 0) {
			usesCache = 0; 
			printf("no_cache option specified, don't use cache.\n");
		}
		else {
			printf("Unsupported option, use cache by default. no_cache option available\n");
		}
	}


	// initialize data

	// initialize array on CPU
	for (int i = 0; i < arraySize; i++) {
		arr[i] = 1;
	}
	*sumValue = 0;

	// Prefetch data to GPU
	int device = -1;
	cudaGetDevice(&device);
	cudaDeviceSynchronize();
	cudaMemPrefetchAsync(arr, arraySize * sizeof(int), device, NULL);
	cudaMemPrefetchAsync(sumValue, sizeof(int), device, NULL);
	cudaDeviceSynchronize();
	//printf("Device used: %d\n", device);


	// Time measurement
	int elapsedClocks = 0, startClock = 0, endClock = 0;
	double elapsedTime, avgElapsedTime;

	startClock = clock();
	if (usesCache) {
		arraySumGPU << <1, 1 >> > (arr, arraySize, sumValue, numCycles);
	}
	else {
		arraySumStrideGPU << <1, 1 >> > (arr, arraySize, sumValue, numCycles, cacheLineSize);
	}
	cudaDeviceSynchronize();
	endClock = clock();

	// Print results 
	printf("startClock: %d\n", startClock);
	printf("endClock: %d\n", endClock);
	elapsedClocks = endClock - startClock;
	printf("elapsedClock: %d\n", elapsedClocks);
	elapsedTime = ((double)(elapsedClocks)) / (CLOCKS_PER_SEC);
	avgElapsedTime = elapsedTime / numCycles;
	printf("Sum= %d. Number of repetitions= %d.\nElapsed time= %fs. Average elapsed time= %fs.\n", *sumValue, numCycles, elapsedTime, avgElapsedTime);


	cudaFree(arr);
	cudaFree(sumValue);
	cudaDeviceReset();
	return 0;
}