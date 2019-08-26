#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <string.h>

#include "anyoption.h" // options parsing

/*
How to compile:
	nvcc "filename" anyoption.cpp
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

// sum array on multiple threads on GPU
__global__ void arraySumGPU(int *arr, int arraySize, int *sumValue, int numCycles) {
	int index = threadIdx.x; // current thread ID
	int numThreads = blockDim.x;
	int numElem = arraySize / numThreads; // # of elements per thread
	int tempSum = 0;
	for (int k = 0; k < numCycles; k++) {
		tempSum = 0;
		for (int i = index * numElem; (i < arraySize) && (i < (index + 1) * numElem); i++) {
			tempSum += arr[i];
		}
	}
	//printf("-tempSum thread %d = %d;\n", index, tempSum);
	*sumValue += tempSum;
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

// allocate memory using "malloc" if on CPU, "cudaMallocManaged" if on GPU
void genericMalloc(void **ptr, int size, int onCPU) {
	if (onCPU) {
		*ptr = malloc(size);
	}
	else {
		cudaMallocManaged(ptr, size);
	}
}

// free memory, using either free or cudaFree
void genericFree(void *ptr, int onCPU) {
	if (onCPU) {
		free(ptr);
	}
	else {
		cudaFree(ptr);
	}
}


int main(int argc, char *argv[])
{
	int arraySize = 1 << 23; // 8M integers
	int usesCache = 1; // 0: don't use cache, 1: use cache (default)
	int onCPU = 0; // allocate memory with CUDA functions
	int cacheLineSize = 64 / sizeof(int); // # integers per cache line on CPU
	int cacheLineSizeGPU = 128 / sizeof(int); // # integers per cache line on GPU
	int numCycles = 500; // default # of repetitions on CPU
	int numCyclesGPU = 30; // default # of repetitions on GPU
	int numThreads = 1; // number of threads running on kernel
	int maxNumThreads = 1024; 
	int *arr, *sumValue;
	printf("Default options: sum on CPU, use cache, arraySize= %d integers.\n", arraySize);
	printf("-- Default number of repetitions: %d (CPU), %d (GPU).\n", numCycles, numCyclesGPU);
	printf("-- max number of threads= %d.\n", maxNumThreads);


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
		numCyclesGPU = numCycles;
		printf("Number of repetitions set to: %d\n", numCycles);
	}
	if (opt->getValue("size") != NULL) {
		arraySize = (1 << 20) * atoi(opt->getValue("size"));
		printf("Array size set to: %dM integers\n", arraySize);
	}

	delete opt;
	// options parsed

	// allocate memory
	genericMalloc((void**)&arr, arraySize * sizeof(int), onCPU);
	genericMalloc((void**)&sumValue, arraySize * sizeof(int), onCPU);

	// initialize array on CPU
	for (int i = 0; i < arraySize; i++) {
		arr[i] = 1;
	}
	*sumValue = 0;


	// Time measurement
	int elapsedClocks = 0, startClock = 0, endClock = 0;
	double elapsedTimeCPU, avgElapsedTimeCPU, elapsedTimeGPU, avgElapsedTimeGPU;

	// Measure CPU execution time
	startClock = clock();
	if (usesCache) {
		arraySum(arr, arraySize, sumValue, numCycles);
	}
	else {
		arraySumStride(arr, arraySize, sumValue, numCycles, cacheLineSize);
	}
	endClock = clock();

	// Print CPU results
	printf("startClock CPU: %d, endClock CPU: %d\n", startClock, endClock);
	elapsedClocks = endClock - startClock;
	printf("elapsedClock CPU: %d\n", elapsedClocks);
	elapsedTimeCPU = ((double)(elapsedClocks)) / (CLOCKS_PER_SEC);
	avgElapsedTimeCPU = elapsedTimeCPU / numCycles;
	printf("CPU: sum= %d. \nElapsed time= %fs. Average execution time= %fs.\n\n", *sumValue, elapsedTimeCPU, avgElapsedTimeCPU);


	// Prefetch data to GPU
	*sumValue = 0;
	int device = -1;
	cudaGetDevice(&device);
	cudaDeviceSynchronize();
	cudaMemPrefetchAsync(arr, arraySize * sizeof(int), device, NULL);
	cudaMemPrefetchAsync(sumValue, sizeof(int), device, NULL);
	cudaDeviceSynchronize();


	// Measure GPU execution time
	avgElapsedTimeGPU = avgElapsedTimeCPU + 10; // run on GPU at least once
	while ((avgElapsedTimeCPU < avgElapsedTimeGPU) && (numThreads <= maxNumThreads)) {
		*sumValue = 0;
		cudaMemPrefetchAsync(sumValue, sizeof(int), device, NULL);
		cudaDeviceSynchronize();

		startClock = clock();
		if (usesCache) {
			arraySumGPU << <1, numThreads >> > (arr, arraySize, sumValue, numCyclesGPU);
		}
		else {
			arraySumStrideGPU << <1, 1 >> > (arr, arraySize, sumValue, numCyclesGPU, cacheLineSizeGPU);
		}
		cudaDeviceSynchronize();
		endClock = clock();

		// Print GPU results
		elapsedClocks = endClock - startClock;
		elapsedTimeGPU = ((double)(elapsedClocks)) / (CLOCKS_PER_SEC);
		avgElapsedTimeGPU = elapsedTimeGPU / numCyclesGPU;
		printf("GPU: %d threads.\n-- Elapsed time= %fs. Average execution time= %fs.\n", numThreads, elapsedTimeGPU, avgElapsedTimeGPU);

		if (avgElapsedTimeCPU < avgElapsedTimeGPU) {
			// double thread numbers
			numThreads *= 2;
		}
	}


	// print comparison results
	if (avgElapsedTimeCPU >= avgElapsedTimeGPU) {
		printf("GPU surpassed CPU when running %d threads\n. CPU time: %f, GPU time: %f.\n\n", numThreads, avgElapsedTimeCPU, avgElapsedTimeGPU);
	}
	else{
		printf("CPU is still faster.\n\n");
	}
	

	// free allocated memory
	genericFree(arr, onCPU);
	genericFree(sumValue, onCPU);

	return 0;
}
