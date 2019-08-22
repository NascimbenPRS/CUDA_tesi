#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>

#include "anyoption.h" // options parsing

/*
How to compile:
	nvcc "filename" anyoption.cpp
*/

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

// allocate memory using "cudaMallocManaged" if compiled with nvcc, "malloc" otherwise
void genericMalloc(void **ptr, int size) {
	// compiler= nvcc
#ifdef __NVCC__
	cudaMallocManaged(ptr, size);
#endif

	// compiler!= nvcc
#ifndef __NVCC__
	*ptr = malloc(size);
#endif
}

// free memory, using either free or cudaFree
void genericFree(void *ptr) {
	// compiler= nvcc
#ifdef __NVCC__
	cudaFree(ptr);
#endif

	// compiler!= nvcc
#ifndef __NVCC__
	free(ptr);
#endif
}


int main(int argc, char *argv[])
{
	printf("\nSum array of integers on GPU (single thread).\n");
#ifdef __NVCC__
	printf("Compiled with nvcc\n");
#endif
#ifndef __NVCC__
	printf("Not compiled with nvcc\n");
#endif

	int arraySize = 1 << 23; // 8M integers
	int usesCache = 1; // 0: don't use cache, 1: use cache (default)
	int cacheLineSize = 128 / sizeof(int); // # integers per cache line
	int numCycles = 40; // # of repetitions
	int *arr, *sumValue;
	printf("Default options: use cache, arraySize= %d integers, %d repetitions.\n", arraySize, numCycles);


	// parse options
	AnyOption *opt = new AnyOption();
	// set usage
	opt->addUsage("Options usage: ");
	opt->addUsage("");
	opt->addUsage(" --no_cache \tDon't use cache ");
	opt->addUsage(" --rep <rep>\tNumber of repetitions ");
	opt->addUsage(" --size <size>\tArray size (* 2^20) elements");
	opt->addUsage("");
	opt->printUsage();

	// set options
	opt->setFlag("no_cache");
	opt->setOption("rep");
	opt->setOption("size");

	// Process commandline and get the options
	opt->processCommandArgs(argc, argv);

	// Get option values
	if (opt->getFlag("no_cache")) {
		usesCache = 0;
		printf("no_cache flag set\n");
	}
	if (opt->getValue("rep") != NULL) {
		numCycles = atoi(opt->getValue("rep"));
		printf("Number of repetitions set to: %d\n", numCycles);
	}
	if (opt->getValue("size") != NULL) {
		arraySize = (1 << 20) * atoi(opt->getValue("size"));
		printf("Array size set to: %dM integers\n", arraySize);
	}

	delete opt;
	// options parsed

	// allocate memory
	genericMalloc((void**)&arr, arraySize * sizeof(int));
	genericMalloc((void**)&sumValue, sizeof(int));

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
	printf("Sum= %d. \nElapsed time= %fs. Average elapsed time= %fs.\n\n", *sumValue, elapsedTime, avgElapsedTime);

	
	// free allocated memory
	genericFree(arr);
	genericFree(sumValue);

	cudaDeviceReset();
	return 0;
}