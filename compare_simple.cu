#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include "commonFunctions.h" // generic malloc
#include "sumOptions.h" // options handling


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
__global__ void arraySumGPU(int **arr, int arraySize, int *sumValue, int numCycles) {
	int index = threadIdx.x; // current thread ID
	int tempSum = 0;
	for (int k = 0; k < numCycles; k++) {
		tempSum = 0;
		for (int i = 0; i < arraySize; i++) {
			tempSum += arr[index][i];
		}
	}
	sumValue[index] = tempSum;
}

// sum array of integers non-sequentially (not using cache) on multiple threads on GPU
__global__ void arraySumStrideGPU(int **arr, int arraySize, int *sumValue, int numCycles, int cacheLineSize) {
	int index = threadIdx.x; // current thread ID
	int tempSum = 0;
	for (int k = 0; k < numCycles; k++) {
		tempSum = 0;
		for (int i = 0; i < cacheLineSize; i++) {
			for (int j = i; j < arraySize; j += cacheLineSize) {
				tempSum += arr[index][j];
			}
		}
	}
	sumValue[index] = tempSum;
}

// initialize array of arrays using multiple threads
__global__ void initializeArraysGPU(int **arrGPU, int arraySize) {
	int index = threadIdx.x; // current thread ID
	for (int i = 0; i < arraySize; i++) {
		arrGPU[index][i] = 1;
	}
}



int main(int argc, char *argv[])
{
	int arraySize = 1 << 20; //
	int usesCache = 1; // 0: don't use cache, 1: use cache (default)
	int cacheLineSize = 64 / sizeof(int); // # integers per cache line on CPU
	int cacheLineSizeGPU = 128 / sizeof(int); // # integers per cache line on GPU
	int numCycles = 1000; // default # of repetitions on CPU
	int numCyclesGPU = 30; // default # of repetitions on GPU
	int *arrCPU, *sumValueCPU, **arrGPU, *sumValuesGPU; // arrGPU: array of arrays
	int numThreads = 1; // # of threads
	int numBlocks = 1; // # of blocks
	int maxNumThreads = 1024;
	int maxNumBlocks= 1024;
	printf("\n Compare execution time on CPU and GPU\n");
	printf("Default options: use cache, arraySize= %d integers.\n", arraySize);
	//printf("-- Default number of repetitions: %d (CPU), %d (GPU).\n", numCycles, numCyclesGPU);
	printf("-- max number of threads= %d.\n", maxNumThreads);
	

	readOptions(argc, argv, &usesCache, &numCycles, &numCyclesGPU, &arraySize); // read options from command line and update values accordingly
	// don't use --rep
	

	// allocate memory for CPU execution
	genericMalloc((void**)&arrCPU, arraySize * sizeof(int));
	genericMalloc((void**)&sumValueCPU, sizeof(int));

	// initialize array on CPU
	for (int i = 0; i < arraySize; i++) {
		arrCPU[i] = 1;
	}
	*sumValueCPU = 0;


	// Time measurement
	int minRunTime = 4; // elapsedTime must be at least minRunTime seconds
	int minNumCyclesCPU = 1, minNumCyclesGPU = 1;
	int elapsedClocks = 0, startClock = 0, endClock = 0;
	double elapsedTimeCPU= 0.f, avgElapsedTimeCPU= 0.f, elapsedTimeGPU= 0.f, avgElapsedTimeGPU= 0.f;


	avgElapsedTimeGPU = avgElapsedTimeCPU + 1; // run at least once
	while ((avgElapsedTimeCPU < avgElapsedTimeGPU) && (numThreads <= maxNumThreads)) {
		printf("NUM THREADS= %d\n", numThreads);
		
		// Measure CPU execution time
		elapsedTimeCPU = 0.f;
		avgElapsedTimeCPU = 0.f;
		numCycles = minNumCyclesCPU;
		while (elapsedTimeCPU < minRunTime) {// double numCycles until execution takes at least minRunTime
			numCycles *= 2;
			startClock = clock();
			if (usesCache) {
				arraySum(arrCPU, arraySize, sumValueCPU, numThreads * numCycles);
			}
			else {
				arraySumStride(arrCPU, arraySize, sumValueCPU, numThreads * numCycles, cacheLineSize);
			}
			endClock = clock();

			elapsedClocks = endClock - startClock;
			elapsedTimeCPU = ((double)(elapsedClocks)) / (CLOCKS_PER_SEC);
		}
		avgElapsedTimeCPU = elapsedTimeCPU / numCycles; // = avg time * numThreads
		printf("CPU: Elapsed time= %fs. Average execution time= %fs.\n", elapsedTimeCPU, avgElapsedTimeCPU);


		// Measure GPU execution time

		// allocate memory for GPU execution
		genericMalloc((void**)&arrGPU, numThreads * sizeof(int*));
		genericMalloc((void**)&sumValuesGPU, numThreads * sizeof(int));
			// allocate for each array copy
		for (int i = 0; i < numThreads; i++) {
			genericMalloc((void**)&arrGPU[i], arraySize * sizeof(int));
		}
			// initialize arrays on GPU
		initializeArraysGPU << <1, numThreads >> > (arrGPU, arraySize);
		cudaDeviceSynchronize();

		elapsedTimeGPU = 0.f;
		avgElapsedTimeGPU = 0.f;
		numCyclesGPU = minNumCyclesGPU;
		while (elapsedTimeGPU < minRunTime) {// double numCyclesGPU until execution takes at least minRunTime
			numCyclesGPU *= 2;
			startClock = clock();
			if (usesCache) {
				arraySumGPU << <1, numThreads >> > (arrGPU, arraySize, sumValuesGPU, numCyclesGPU);
			}
			else {
				arraySumStrideGPU << <1, numThreads >> > (arrGPU, arraySize, sumValuesGPU, numCyclesGPU, cacheLineSizeGPU);
			}
			cudaDeviceSynchronize();
			endClock = clock();

			elapsedClocks = endClock - startClock;
			elapsedTimeGPU = ((double)(elapsedClocks)) / (CLOCKS_PER_SEC);
			
			//printf("----elapsedGPUTIME: %f\n", elapsedTimeGPU);
		}

		avgElapsedTimeGPU = elapsedTimeGPU / numCyclesGPU;
		printf("GPU: Elapsed time= %fs. Average execution time= %fs.\n\n", elapsedTimeGPU, avgElapsedTimeGPU);
		
		// Free GPU memory
		genericFree(sumValuesGPU);
		for (int i = 0; i < numThreads; i++) {
			genericFree(arrGPU[i]);
		}
		genericFree(arrGPU);
		if (avgElapsedTimeCPU < avgElapsedTimeGPU) {
			numThreads *= 2;
		}

	}


	// print comparison results
	if (avgElapsedTimeCPU >= avgElapsedTimeGPU) {
		printf("GPU surpassed CPU when running %d threads\n. CPU time: %f, GPU time: %f.\n\n", numThreads, avgElapsedTimeCPU, avgElapsedTimeGPU);
	}
	else {
		printf("CPU is still faster.\n\n");
	}


	// free allocated memory for CPU
	genericFree(arrCPU);
	genericFree(sumValueCPU);

	return 0;
}
