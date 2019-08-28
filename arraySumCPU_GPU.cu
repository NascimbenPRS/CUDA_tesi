#ifdef __NVCC__
	#include "cuda_runtime.h"
	
	//#include "device_launch_parameters.h"
#endif


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include "commonFunctions.h" // generic malloc
#include "sumOptions.h" // options handling


/*
How to compile:
	nvcc/g++ "filename" anyoption.cpp
*/


// sum array of integers sequentially (using cache)
void arraySum(int *arr, int arraySize, int *sumValue, int numCycles) {
	int tempSum = 0;
	for (int k = 0; k < numCycles; k++) {
		tempSum = 0;
		for (int i = 0; i < arraySize; i++) {
			tempSum += arr[i];
		}
	}
	*sumValue = tempSum;
}

// sum array of integers non-sequentially (not using cache)
void arraySumStride(int *arr, int arraySize, int *sumValue, int numCycles, int cacheLineSize) {
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

// GPU kernels
#ifdef __NVCC__ 
// sum array of integers sequentially (using cache) on GPU
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

// sum array of integers non-sequentially (not using cache) on GPU
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
#endif


int main(int argc, char *argv[])
{
	int arraySize = 1 << 23; // 8M integers
	int usesCache = 1; // 0: don't use cache, 1: use cache (default)
	int cacheLineSize = 64 / sizeof(int); // # integers per cache line on CPU
	int cacheLineSizeGPU = 128 / sizeof(int); // # integers per cache line on GPU
	int numCycles = 1000; // default # of repetitions on CPU
	int numCyclesGPU = 30; // default # of repetitions on GPU
	int *arr, *sumValue;
	printf("Default options: use cache, arraySize= %d integers.\n", arraySize);
	printf("-- Default number of repetitions: %d (CPU), %d (GPU).\n", numCycles, numCyclesGPU);

	// read options from command line and update values accordingly
	readOptions(argc, argv, &usesCache, &numCycles, &numCyclesGPU, &arraySize); 


	// allocate memory
	genericMalloc((void**)&arr, arraySize * sizeof(int));
	genericMalloc((void**)&sumValue, arraySize * sizeof(int));

	// initialize array on CPU
	for (int i = 0; i < arraySize; i++) {
		arr[i] = 1;
	}
	*sumValue = 0;


	// Time measurement
	int elapsedClocks = 0, startClock = 0, endClock = 0;
	double elapsedTime, avgElapsedTime;


#ifndef __NVCC__
	printf("Not compiled with NVCC, run on CPU\n");
	startClock = clock();
	if (usesCache) {
		arraySum(arr, arraySize, sumValue, numCycles);
	}
	else {
		arraySumStride(arr, arraySize, sumValue, numCycles, cacheLineSize);
	}
#endif
#ifdef __NVCC__
	printf("Compiled with NVCC, run on GPU\n");
	// Prefetch data to GPU
	int device = -1;
	cudaGetDevice(&device);
	cudaDeviceSynchronize();
	cudaMemPrefetchAsync(arr, arraySize * sizeof(int), device, NULL);
	cudaMemPrefetchAsync(sumValue, sizeof(int), device, NULL);
	cudaDeviceSynchronize();

	startClock = clock();
	if (usesCache) {
		arraySumGPU << <1, 1 >> > (arr, arraySize, sumValue, numCyclesGPU);
	}
	else {
		arraySumStrideGPU << <1, 1 >> > (arr, arraySize, sumValue, numCyclesGPU, cacheLineSizeGPU);
	}
	cudaDeviceSynchronize();
#endif
	endClock = clock();


	// Print results
	printf("startClock: %d, endClock: %d\n", startClock, endClock);
	elapsedClocks = endClock - startClock;
	printf("elapsedClock: %d\n", elapsedClocks);
	elapsedTime = ((double)(elapsedClocks)) / (CLOCKS_PER_SEC);
#ifndef __NVCC__
	avgElapsedTime = elapsedTime / numCycles;
#endif
#ifdef __NVCC__
	avgElapsedTime = elapsedTime / numCyclesGPU;
#endif
	printf("Sum= %d. \nElapsed time= %fs. Average execution time= %fs.\n\n", *sumValue, elapsedTime, avgElapsedTime);


	// free allocated memory
	genericFree(arr);
	genericFree(sumValue);

	return 0;
}
