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
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
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
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
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


// struct to keep track of performances
struct performance
{
	int numThreads= 0;
	int numBlocks= 0;
	int numThreadsTotal= 0;
	double timeCPU= 0.f;
	double timeGPU= 0.f;
	double speedup= 0.f;
};


void printBestPerformances(struct performance *bestPerf, int numSavedPerformances) {
	printf("\nBest %d performances:\n", numSavedPerformances);
	for (int i = 0; i < numSavedPerformances; i++) {
		printf("%d) %d blocks, %d threads per block, %d threads running\n", i+1, bestPerf[i].numBlocks, bestPerf[i].numThreads, bestPerf[i].numThreadsTotal);
		printf("-- CPU time: %f, GPU time: %f, speedup= %f\n", bestPerf[i].timeCPU, bestPerf[i].timeGPU, bestPerf[i].speedup);
	}
	printf("-------------\n");
}

void updateBestPerformances(struct performance *bestPerf, int numSavedPerformances, struct performance perf) {
	int i = 0;
	while (i < numSavedPerformances) {
		if (perf.speedup > bestPerf[i].speedup) {
			// shift performances
			for (int k = numSavedPerformances - 1; k > i; k--) {
				bestPerf[k] = bestPerf[k - 1];
			}
			bestPerf[i] = perf;
			i = numSavedPerformances; // force exit
		}
		else {
			i++;
		}
	}
}

// print the best performance running searchedNumThreads concurrent threads in bestPerf
void printBestPerformanceNumThreads(struct performance *bestPerf, int numSavedPerformances, int searchedNumThreads) {
	int i = 0;
	while (i < numSavedPerformances) {
		if (bestPerf[i].numThreadsTotal == searchedNumThreads) {
			printf("Running threads: %d\n", searchedNumThreads);
			printf("-- %d blocks, %d threads per block\n", bestPerf[i].numBlocks, bestPerf[i].numThreads);
			printf("-- CPU time: %f, GPU time: %f, speedup= %f\n\n", bestPerf[i].timeCPU, bestPerf[i].timeGPU, bestPerf[i].speedup);
			i = numSavedPerformances; // force exit
		}
		else {
			i++;
		}
	}
}




/*
Ranks all <<< gridSize, blockSize>>> GPU kernel configurations, where:
-- gridSize <= maxNumBlocks;
-- (gridSize x blocksize) <= maxNumThreads.

Configurations are ranked by speedup (i.e. CPU_Time/GPU_Time), where:
-- CPU_Time= time to sum numThreadsTotal arrays of integers on CPU (serially);
-- GPU_Time= time to sum numThreadsTotal array of integers on GPU, using a certain kernel configuration.


Shows the best configuration for each possible number of threads (1, 2, .., maxNumThreads)
*/

int main(int argc, char *argv[])
{
	int arraySize = 1 << 20; // 1M integers
	int usesCache = 1; // 0: don't use cache, 1: use cache (default)
	int cacheLineSize = 64 / sizeof(int); // # integers per cache line on CPU
	int cacheLineSizeGPU = 128 / sizeof(int); // # integers per cache line on GPU
	int numCycles = 1000; // default # of repetitions on CPU
	int numCyclesGPU = 30; // default # of repetitions on GPU
	int *arrCPU, *sumValueCPU, **arrGPU, *sumValuesGPU; // arrGPU: array of arrays
	int numThreads = 1; // # of threads per block
	int numBlocks = 1; // # of blocks
	int numThreadsTotal; // # of threads running concurrently (= numThreads * numBlocks)
	int maxNumThreads = 1024; // max # of threads running concurrently
	int maxNumBlocks = 512; // max # of usable blocks
	
	// create an array to save performances
	int numSavedPerformances = 200; // # best performances to record
	struct performance *bestPerformances;
	genericMalloc((void**)&bestPerformances, numSavedPerformances * sizeof(struct performance));



	cudaDeviceProp prop;
	int device = -1;
	cudaGetDevice(&device);
	cudaDeviceSynchronize();
	cudaGetDeviceProperties(&prop, device);
	int numSM = prop.multiProcessorCount;

	printf("\nCompare execution time on CPU and GPU\n");
	printf("Default options: use cache, arraySize= %d integers.\n", arraySize);
	//printf("-- Default number of repetitions: %d (CPU), %d (GPU).\n", numCycles, numCyclesGPU);
	printf("-- number of multiprocessors on GPU= %d\n", numSM);
	printf("-- max number of blocks= %d.\n", maxNumBlocks);
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
	int minRunTime = 5; // elapsedTime must be at least minRunTime seconds
	int minNumCyclesCPU = 1, minNumCyclesGPU = 1;
	int elapsedClocks = 0, startClock = 0, endClock = 0;
	double elapsedTimeCPU = 0.f, avgElapsedTimeCPU = 0.f, elapsedTimeGPU = 0.f, avgElapsedTimeGPU = 0.f;
	double speedup = 0.f; // CPU time/GPU time


	while (numBlocks <= maxNumBlocks) {
		numThreadsTotal = numBlocks * numThreads;
		printf("NUM BLOCKS= %d, NUM THREADS PER BLOCK = %d, TOTAL NUM THREADS= %d\n", numBlocks, numThreads, numThreadsTotal);
		// Measure CPU execution time
		elapsedTimeCPU = 0.f;
		avgElapsedTimeCPU = 0.f;
		numCycles = minNumCyclesCPU;
		while (elapsedTimeCPU < minRunTime) {// double numCycles until execution takes at least minRunTime
			numCycles *= 2;
			startClock = clock();
			if (usesCache) {
				arraySum(arrCPU, arraySize, sumValueCPU, numThreadsTotal * numCycles);
			}
			else {
				arraySumStride(arrCPU, arraySize, sumValueCPU, numThreadsTotal * numCycles, cacheLineSize);
			}
			endClock = clock();

			elapsedClocks = endClock - startClock;
			elapsedTimeCPU = ((double)(elapsedClocks)) / (CLOCKS_PER_SEC);
		}
		avgElapsedTimeCPU = elapsedTimeCPU / numCycles; // = avg time * numThreadsTotal
		//printf("--CPU: Elapsed time= %fs. \n", elapsedTimeCPU);
		printf("-- CPU: Average execution time= %fs.\n", avgElapsedTimeCPU);


		// Measure GPU execution time

		// allocate memory for GPU execution
		genericMalloc((void**)&arrGPU, numThreadsTotal * sizeof(int*));
		genericMalloc((void**)&sumValuesGPU, numThreadsTotal * sizeof(int));
		// allocate for each array copy
		for (int i = 0; i < numThreadsTotal; i++) {
			genericMalloc((void**)&arrGPU[i], arraySize * sizeof(int));
		}
		// initialize arrays on GPU
		initializeArraysGPU << <1, numThreadsTotal >> > (arrGPU, arraySize);
		cudaDeviceSynchronize();

		elapsedTimeGPU = 0.f;
		avgElapsedTimeGPU = 0.f;
		numCyclesGPU = minNumCyclesGPU;
		while (elapsedTimeGPU < minRunTime) {// double numCyclesGPU until execution takes at least minRunTime
			numCyclesGPU *= 2;
			startClock = clock();
			if (usesCache) {
				arraySumGPU << <numBlocks, numThreads >> > (arrGPU, arraySize, sumValuesGPU, numCyclesGPU);
			}
			else {
				arraySumStrideGPU << <numBlocks, numThreads >> > (arrGPU, arraySize, sumValuesGPU, numCyclesGPU, cacheLineSizeGPU);
			}
			cudaDeviceSynchronize();
			endClock = clock();

			elapsedClocks = endClock - startClock;
			elapsedTimeGPU = ((double)(elapsedClocks)) / (CLOCKS_PER_SEC);
		}

		avgElapsedTimeGPU = elapsedTimeGPU / numCyclesGPU;
		speedup = avgElapsedTimeCPU / avgElapsedTimeGPU;
		// printf("--GPU: Elapsed time= %fs. \n", elapsedTimeGPU);
		printf("-- GPU: Average execution time= %fs.\n", avgElapsedTimeGPU);
		printf("--speedup : %f\n\n", speedup);

		// update record of best performances
		struct performance current_perf = { numThreads, numBlocks, numThreadsTotal, avgElapsedTimeCPU, avgElapsedTimeGPU, speedup };
		updateBestPerformances(bestPerformances, numSavedPerformances, current_perf);

		// Free GPU memory
		genericFree(sumValuesGPU);
		for (int i = 0; i < numThreadsTotal; i++) {
			genericFree(arrGPU[i]);
		}
		genericFree(arrGPU);

		if (numThreadsTotal >= maxNumThreads) {
			// double numBlocks and reset numThreads
			numBlocks *= 2;
			numThreads = 1;
			printf("-------------------\n\n");
		}
		else {
			numThreads *= 2;
		}
	}


	// Print overall results
	printBestPerformances(bestPerformances, numSavedPerformances);
	// Print best performance for each possible number of threads
	for (int i = 1; i <= maxNumThreads; i *= 2) {
		printBestPerformanceNumThreads(bestPerformances, numSavedPerformances, i);
	}
	genericFree(bestPerformances);
	

	// free allocated memory for CPU
	genericFree(arrCPU);
	genericFree(sumValueCPU);

	

	return 0;
}
