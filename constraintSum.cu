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


// sum integers dataArr[j], where j belongs to a constraint (array of indexes)
void constraintSum(int *dataArr, int arraySize, int *sumValue, int numCycles, int *constraint, int constraintSize) {
	int tempSum = 0;
	for (int k = 0; k < numCycles; k++) {
		tempSum = 0;
		for (int i = 0; i < constraintSize; i++) {
			tempSum += dataArr[constraint[i]];
		}
	}
	*sumValue = tempSum;
}

// sum integers dataArr[j], where j belongs to a constraint (array of indexes) on GPU
__global__ void constraintSumGPU(int *dataArr, int arraySize, int *sumValues, int numCycles, int numConstraints, int **constraints, int *constraintSizes) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index <= numConstraints) {
		// constraintArr[index] is a valid constraint
		int tempSum = 0;
		for (int k = 0; k < numCycles; k++) {
			tempSum = 0;
			for (int i = 0; i < constraintSizes[index]; i++) {
				tempSum += dataArr[constraints[index][i]];
			}
		}
		sumValues[index] = tempSum;
	}
	else {
		// invalid constraint index
	}
}

// generate random integer in {min,..,max}
int randomIntRange(int min, int max) {
	return min + rand() / (RAND_MAX / (max - min + 1) + 1);
}

// fill a constraint with all-different random integers between 0 and arraySize-1
void fillConstraintRandom(int *constraint, int constraintSize, int arraySize) {
	int temp;
	int k = 0;
	bool isNew = true;
	bool foundNew = false;
	for (int i = 0; i < constraintSize; i++) {
		foundNew = false;
		while (!foundNew) {
			// try new random value
			temp = randomIntRange(0, arraySize - 1);
			isNew = true;
			k = 0;
			while ((k < i) && (isNew)) {
				// compare to previous entries
				if (constraint[k] == temp) {
					isNew = false;
				}
				k++;
			}
			if (isNew) {
				constraint[i] = temp;
				foundNew = true; // allows for-cycle to proceed
			}
		}
	}
}

// get the optimal <<<numOfBlocks, blockSize>>> configuration, given the number of jobs and of multiprocessors available on GPU
void getOptimalGridConfig(int numOfJobs, int numOfMultiProcessors, int maxThreadsPerBlock, int *numOfBlocks, int *blockSize) {
	if (numOfJobs <= numOfMultiProcessors){
		*numOfBlocks = numOfJobs;
		*blockSize = 1;
	}
	else if (numOfJobs <= numOfMultiProcessors * maxThreadsPerBlock){
		*numOfBlocks = numOfMultiProcessors;
		*blockSize = ceil(*numOfBlocks / numOfMultiProcessors);
	}
	else{
		*numOfBlocks = ceil(numOfJobs / maxThreadsPerBlock);
		*blockSize = maxThreadsPerBlock;
	}
}


/*
// initialize array of arrays using multiple threads
__global__ void initializeArraysGPU(int **arrGPU, int arraySize) {
	int index = threadIdx.x; // current thread ID
	for (int i = 0; i < arraySize; i++) {
		arrGPU[index][i] = 1;
	}
}
*/



int main(int argc, char *argv[])
{
	int arraySize = 1 << 20; // 1M integers
	/* VARIABILI INUTILIZZATE
	int usesCache = 1; // 0: don't use cache, 1: use cache (default)
	int cacheLineSize = 64 / sizeof(int); // # integers per cache line on CPU
	int cacheLineSizeGPU = 128 / sizeof(int); // # integers per cache line on GPU
	*/
	int numCycles = 1000; // default # of repetitions on CPU
	int numCyclesGPU = 30; // default # of repetitions on GPU
	int *dataArray, *resultsCPU, *resultsGPU;
	int blockSize = 1; // # of threads per block
	int numBlocks = 1; // # of blocks
	int numThreadsTotal; // # of threads running concurrently (= numThreads * numBlocks)
	int maxBlockSize = 1024;
	int maxNumBlocks = 1024;

	// Constraint variables
	int a = 2 << 10; // # constraints of size 2 (2 variables involved, e.g. x1 + x2 = 0)
	int b = 2 << 10; // # constraints of size 3
	int c = 2 << 10; // # constraints of size in {4..128}
	int numConstraints = a + b + c; // # all constraints
	int *constraintSizes; // constraintSizes[i] == size of constraint i
	int **constraints; // array of constraints
	// set constraint sizes
	genericMalloc((void**)&constraintSizes, numConstraints * sizeof(int));
	for (int i = 0; i < numThreadsTotal; i++) {
		if (i < a) {
			constraintSizes[i] = 2;
		}
		else {
			if (i < a + b) {
				constraintSizes[i] = 3;
			}
			else {
				constraintSizes[i] = randomIntRange(4, 128); // random size 4..128
			}
		}
	}
	// allocate constraints array
	genericMalloc((void**)&constraints, numConstraints * sizeof(int*));
	for (int i = 0; i < numConstraints; i++) {
		genericMalloc((void**)&constraints[i], constraintSizes[i] * sizeof(int));
	}
	// initialize constraints with random values
	for (int i = 0; i < numConstraints; i++) {
		fillConstraintRandom(constraints[i], constraintSizes[i], arraySize);
	}
	// allocate and initialize CPU and GPU results arrays
	genericMalloc((void**)&resultsCPU, numConstraints * sizeof(int));
	genericMalloc((void**)&resultsGPU, sizeof(int));
	for (int i = 0; i < numConstraints; i++) {
		resultsCPU[i] = 0;
		resultsGPU[i] = 0;
	}
	
	
	cudaDeviceProp prop;
	int device = -1;
	cudaGetDevice(&device);
	cudaDeviceSynchronize();
	cudaGetDeviceProperties(&prop, device);
	int numOfMultiProcessors = prop.multiProcessorCount;

	printf("\nSimulate work on constraints using arrays of integers\n");
	printf("Default options: arraySize= %d integers.\n", arraySize);
	printf("-- number of multiprocessors= %d\n", numOfMultiProcessors);

	/* DA AGGIORNARE CON OPZIONI PER CONSTRAINTS (a,b,c,numConstraints ecc.)
	readOptions(argc, argv, &usesCache, &numCycles, &numCyclesGPU, &arraySize); // read options from command line and update values accordingly
	// don't use --rep
	*/

	// allocate and initialize data array
	genericMalloc((void**)&dataArray, arraySize * sizeof(int));
	for (int i = 0; i < arraySize; i++) {
		dataArray[i] = 1;
	}

	// get optimal GPU grid configuration
	getOptimalGridConfig(numConstraints, numOfMultiProcessors, maxBlockSize, &numBlocks, &blockSize);

	// Time measurement
	int minRunTime = 10; // elapsedTime must be at least minRunTime seconds
	int minNumCyclesCPU = 1, minNumCyclesGPU = 1;
	int elapsedClocks = 0, startClock = 0, endClock = 0;
	double elapsedTimeCPU = 0.f, avgElapsedTimeCPU = 0.f, elapsedTimeGPU = 0.f, avgElapsedTimeGPU = 0.f;

	// Measure CPU execution time
	elapsedTimeCPU = 0.f;
	avgElapsedTimeCPU = 0.f;
	numCycles = minNumCyclesCPU;
	while (elapsedTimeCPU < minRunTime) {// double numCycles until execution takes at least minRunTime
		numCycles *= 2;
		startClock = clock();
		for (int i = 0; i < numConstraints; i++) {
			constraintSum(dataArray, arraySize, &resultsCPU[i], numCycles, constraints[i], constraintSizes[i]);
		}
		endClock = clock();
		elapsedClocks = endClock - startClock;
		elapsedTimeCPU = ((double)(elapsedClocks)) / (CLOCKS_PER_SEC);
	}
	avgElapsedTimeCPU = elapsedTimeCPU / numCycles;
	printf("CPU: Elapsed time= %fs. Average execution time= %fs.\n", elapsedTimeCPU, avgElapsedTimeCPU);
	
	// Measure GPU execution time
	elapsedTimeGPU = 0.f;
	avgElapsedTimeGPU = 0.f;
	numCyclesGPU = minNumCyclesGPU;
	cudaMemPrefetchAsync(dataArray, arraySize * sizeof(int), device);
	/*PREFETCH OF CONSTRAINTS AND RESULTSGPU
	*
	*
	*/
	while (elapsedTimeGPU < minRunTime) {// double numCyclesGPU until execution takes at least minRunTime
		numCyclesGPU *= 2;
		startClock = clock();
		constraintSumGPU << <numBlocks, blockSize >> > (dataArray, arraySize, resultsGPU, numCyclesGPU, numConstraints, constraints, constraintSizes );
		cudaDeviceSynchronize();
		endClock = clock();

		elapsedClocks = endClock - startClock;
		elapsedTimeGPU = ((double)(elapsedClocks)) / (CLOCKS_PER_SEC);
	}
	avgElapsedTimeGPU = elapsedTimeGPU / numCyclesGPU;
	printf("GPU: Elapsed time= %fs. Average execution time= %fs.\n\n", elapsedTimeGPU, avgElapsedTimeGPU);

	/* Print comparison results
	*
	*
	*
	*
	*/

	// free allocated memory
	genericFree(dataArray);
	genericFree(resultsCPU);
	genericFree(resultsGPU);
	// FREE CONSTRAINT SIZES AND CONSTRAINTS
	genericFree(constraintSizes);
	for (int i = 0; i < numConstraints; i++) {
		genericFree(constraints[i]);
	}
	genericFree(constraints);

	return 0;
}
