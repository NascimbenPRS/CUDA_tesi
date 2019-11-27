#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include "commonFunctions.h" // generic malloc
#include "constraintsOptions.h" // constraints options handling


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
	if (index < numConstraints) {
		// constraints[index] is a valid constraint
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

// sum integers dataArr[j], where j belongs to a constraint (array of indexes) on GPU.
// Propagate only constraints of size == constLength
__global__ void constraintSumGPU_fixedLength(int *dataArr, int arraySize, int *sumValues, int numCycles, int numConstraints, int **constraints, int *constraintSizes, int constLength) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index < numConstraints) {
		// constraints[index] is a valid constraint
		if (constraintSizes[index] == constLength) {
			int tempSum = 0;
			for (int k = 0; k < numCycles; k++) {
				tempSum = 0;
				for (int i = 0; i < constLength; i++) {
					tempSum += dataArr[constraints[index][i]];
				}
			}
			sumValues[index] = tempSum;
		}
		else {
			// constraintSizes[index] != constLength, do nothing.
		}
	}
	else {
		// invalid constraint index
	}
}

// sum integers dataArr[j], where j belongs to a constraint (array of indexes) on GPU.
// Propagate only constraints of size == constLength
// constraintsFixedSize= array containing indexes of constraints of size constLength; sizeAmount= its cardinality
// 
__global__ void constraintSumGPU_fixedLengthArrays(int *dataArr, int arraySize, int *sumValues, int numCycles, int **constraints, int *constraintsFixedSize, int sizeAmount, int constLength) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index < sizeAmount) {
		// constraintsFixedLength[constLength][index] is a valid constraint (sizeAmount= number of constraints of size constLength
		int tempSum = 0;
		int currentConstraint = constraintsFixedSize[index]; // index of the constraint propagated by this thread
		for (int k = 0; k < numCycles; k++) {
			tempSum = 0;
			for (int i = 0; i < constLength; i++) {
				tempSum += dataArr[constraints[currentConstraint][i]];
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

// Fill a constraint with (all-different) random integers between 0 and arraySize-1
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


// Fill a constraint with the integer sequence [first, first + constraintsSize - 1] mod arraySize;
void fillConstraint(int *constraint, int constraintSize, int arraySize, int first) {
	for (int i = 0; i < constraintSize; i++) {
		constraint[i] = (first + i) % arraySize;
	}
}

// get the optimal <<<numOfBlocks, blockSize>>> configuration, given the number of jobs and multiprocessors available on GPU
void getOptimalGridConfig(int numOfJobs, int numOfMultiProcessors, int maxThreadsPerBlock, int *numOfBlocks, int *blockSize) {
	if (numOfJobs <= numOfMultiProcessors) {
		*numOfBlocks = numOfJobs;
		*blockSize = 1;
	}
	else {
		if (numOfJobs <= (numOfMultiProcessors * maxThreadsPerBlock)) {
			*numOfBlocks = numOfMultiProcessors;
			*blockSize = ceil(((double)numOfJobs) / *numOfBlocks);
		}
		else {
			*numOfBlocks = ceil(((double)numOfJobs) / maxThreadsPerBlock);
			*blockSize = maxThreadsPerBlock;
		}
	}
}




int main(int argc, char *argv[])
{


	int usesCache = 1; // 0: don't use cache, 1: use cache (default)
	/* Unused variables
	int cacheLineSize = 64 / sizeof(int); // # integers per cache line on CPU
	int cacheLineSizeGPU = 128 / sizeof(int); // # integers per cache line on GPU
	*/
	int arraySize = 100; // size of data array
	int numCyclesCPU = 1; // # of repetitions on CPU
	int numCyclesGPU = 1; // # of repetitions on GPU
	int *dataArray, *resultsCPU, *resultsGPU;
	int blockSize = 1; // # of threads per block
	int numBlocks = 1; // # of blocks
	int maxBlockSize = 1024; // max # of threads per block

	// Constraint variables
	int a = 0; // # constraints of size 2 (2 variables involved, e.g. x1 + x2 = 0)
	int b = 0; // # constraints of size 3
	int c = 0; // # constraints of size in {4..128}
	int numConstraints; // # all constraints
	int *constraintSizes; // constraintSizes[i] == size of constraint i
	int **constraints; // array of constraints
	int seed = 113; // seed for the random functions


	// Get device info
	cudaDeviceProp prop;
	int device = -1;
	cudaGetDevice(&device);
	cudaDeviceSynchronize();
	cudaGetDeviceProperties(&prop, device);
	int numOfMultiProcessors = prop.multiProcessorCount;
	// Print default settings
	printf("\nSimulate work on constraints using arrays of integers.\n");
	printf("\nSeparate execution time by constraint size.\n");
	printf("-- number of multiprocessors= %d\n", numOfMultiProcessors);
	printf("Default values: arraySize= %d, a= %d, b= %d, c= %d, seed= %d.\n", arraySize, a, b, c, seed);

	// check arraysize and number of constraints options from command line and update values;
	readOptionsConstraints(argc, argv, &usesCache, &arraySize, &a, &b, &c, &seed, &numCyclesCPU);
	numConstraints = a + b + c;
	// get optimal GPU grid configuration
	getOptimalGridConfig(numConstraints, numOfMultiProcessors, maxBlockSize, &numBlocks, &blockSize);

	// allocate and initialize data array
	genericMalloc((void**)&dataArray, arraySize * sizeof(int));
	for (int i = 0; i < arraySize; i++) {
		dataArray[i] = 1;
	}
	// allocate and initialize constraint sizes
	srand(seed); // set seed for rand()
	int maxConstraintSize = min(arraySize, 128); // prevents constraint size from being > than arraySize
	genericMalloc((void**)&constraintSizes, numConstraints * sizeof(int));
	for (int i = 0; i < numConstraints; i++) {
		if (i < a) {
			constraintSizes[i] = 2;
		}
		else {
			if (i < a + b) {
				constraintSizes[i] = 3;
			}
			else {
				constraintSizes[i] = randomIntRange(4, maxConstraintSize); // random size 4..128 (or arraySize)

			}
		}
	}

	// allocate constraints array
	genericMalloc((void**)&constraints, numConstraints * sizeof(int*));
	for (int i = 0; i < numConstraints; i++) {
		genericMalloc((void**)&constraints[i], constraintSizes[i] * sizeof(int));
	}
	// initialize constraints with random values in [0, arraySize - 1]
	for (int i = 0; i < numConstraints; i++) {
		fillConstraintRandom(constraints[i], constraintSizes[i], arraySize);
	}

	printf("Constraints initialized\n");

	/*
	// PRINT ALL CONSTRAINTS
	for (int i = 0; i < numConstraints; i++) {
		printf("Constraint %d: ", i);
		for (int j = 0; j < constraintSizes[i]; j++) {
			printf("%d,", constraints[i][j]);
		}
		printf("\n");
	}
	*/



	// allocate and initialize CPU and GPU results arrays
	genericMalloc((void**)&resultsCPU, numConstraints * sizeof(int));
	genericMalloc((void**)&resultsGPU, numConstraints * sizeof(int));
	for (int i = 0; i < numConstraints; i++) {
		resultsCPU[i] = 0;
		resultsGPU[i] = 0;
	}

	// Time measurement

	int elapsedClocks = 0, startClock = 0, endClock = 0;
	double elapsedTimeCPU = 0.f, avgElapsedTimeCPU = 0.f, elapsedTimeGPU = 0.f, avgElapsedTimeGPU = 0.f, tempTime = 0.f;
	double normalizedTime = 0.f; // execution time / (constLength * sizesAmount[constLength])

	int** constraintsOfSize; // constraintsOfSize[i]= array containing the indexes of all array of size i
	genericMalloc((void**)&constraintsOfSize, (maxConstraintSize + 1) * sizeof(int*));
	int *sizesAmount; // sizesAmount[i]= # constraints of size i
	genericMalloc((void**)&sizesAmount, (maxConstraintSize + 1) * sizeof(int));
	for (int i = 0; i < maxConstraintSize + 1; i++) {
		sizesAmount[i] = 0;
	}
	for (int i = 0; i < numConstraints; i++) {
		sizesAmount[constraintSizes[i]]++;
	}
	for (int i = 0; i < maxConstraintSize + 1; i++) {
		genericMalloc((void**)&constraintsOfSize[i], sizesAmount[i] * sizeof(int));
		for (int j = 0; j < sizesAmount[i]; j++) {
			constraintsOfSize[i][j] = -1; //initialize with non-valid constraint indexes
		}
	}
	int tempCounter[129];
	for (int i = 0; i < 129; i++) {
		tempCounter[i] = 0;
	}
	for (int i = 0; i < numConstraints; i++) {
		constraintsOfSize[constraintSizes[i]][tempCounter[constraintSizes[i]]] = i;
		tempCounter[constraintSizes[i]]++;
	}

	// Measure CPU execution time
	startClock = clock();
	for (int i = 0; i < numConstraints; i++) {
		constraintSum(dataArray, arraySize, &resultsCPU[i], numCyclesCPU, constraints[i], constraintSizes[i]);
	}
	endClock = clock();
	elapsedClocks = endClock - startClock;
	elapsedTimeCPU = ((double)(elapsedClocks)) / (CLOCKS_PER_SEC);
	avgElapsedTimeCPU = elapsedTimeCPU / numCyclesCPU;




	// Measure GPU execution time
	//		Prefetch data, constraints and resultsGPU arrays to GPU
	cudaMemPrefetchAsync(dataArray, arraySize * sizeof(int), device);
	cudaMemPrefetchAsync(constraints, numConstraints * sizeof(int*), device, NULL);
	for (int i = 0; i < numConstraints; i++) {
		cudaMemPrefetchAsync(constraints[i], constraintSizes[i] * sizeof(int), device, NULL);
	}
	cudaMemPrefetchAsync(constraintSizes, numConstraints * sizeof(int), device, NULL);
	cudaMemPrefetchAsync(resultsGPU, numConstraints * sizeof(int), device, NULL);
	cudaDeviceSynchronize();

	// Create a stream for each constraint size (for each kernel)
	const int numStreams = maxConstraintSize + 1; // a stream for each size in [0, maxConstraintSize]
	cudaStream_t streams[129];
	for (int i = 0; i < numStreams; i++) {
		cudaStreamCreate(&streams[i]);
	}

	int numBlocksArray[129];
	int blockSizeArray[129];
	for (int i = 0; i < maxConstraintSize + 1; i++) {
		getOptimalGridConfig(sizesAmount[i], numOfMultiProcessors, maxBlockSize, &numBlocksArray[i], &blockSizeArray[i]);
	}

	numCyclesGPU = numCyclesCPU;
	//		Propagate constraints of each size in [4, maxConstraintSize] (launching maxConstraintSize+1 concurrent kernels)
	startClock = clock();
	// launch all kernels
	for (int constLength = 4; constLength <= maxConstraintSize; constLength++) {
		constraintSumGPU_fixedLengthArrays << <numBlocksArray[constLength], blockSizeArray[constLength], 0, streams[constLength] >> > (dataArray, arraySize, resultsGPU, numCyclesGPU, constraints, constraintsOfSize[constLength], sizesAmount[constLength],constLength);
	}
	cudaDeviceSynchronize();
	endClock = clock();

	elapsedClocks = endClock - startClock;
	tempTime = ((double)(elapsedClocks)) / (CLOCKS_PER_SEC);
	//normalizedTime = tempTime / (constLength * sizesAmount[constLength]);
	//printf("ConstrSize= %d:\tnumConstr= %d,\tExec time= %f,\tNormalized time= %f;\n", constLength, sizesAmount[constLength], tempTime, normalizedTime);
	elapsedTimeGPU += tempTime;

	avgElapsedTimeGPU = elapsedTimeGPU / numCyclesGPU;
	printf("\nCPU: Elapsed time= %fs. Average execution time= %fs.\n", elapsedTimeCPU, avgElapsedTimeCPU);
	printf("GPU: Total elapsed time= %fs. Average execution time= %fs.\n", elapsedTimeGPU, avgElapsedTimeGPU);


	// Check sizesAmount correctness
	int amountsSum = 0;
	for (int i = 0; i < maxConstraintSize + 1; i++) {
		amountsSum += sizesAmount[i];
	}
	if (amountsSum != numConstraints) {
		printf("ERROR IN SIZES AMOUNT\n");
	}


	// Check if CPU and GPU results match
	bool sameResults = true;
	int j = 0;
	while ((sameResults) && (j < numConstraints)) {
		if (resultsCPU[j] != resultsGPU[j]) {
			sameResults = false;
		}
		j++;
	}

	if (sameResults) {
		printf("Same results on CPU and GPU\n\n");
	}
	else {
		j--;
		printf("Different results on CPU and GPU\n");
		printf("CPU[%d]= %d, GPU[%d]= %d.\n\n", j, resultsCPU[j], j, resultsGPU[j]);
	}



	// Free allocated memory
	genericFree(dataArray);
	genericFree(resultsCPU);
	genericFree(resultsGPU);
	genericFree(constraintSizes);
	for (int i = 0; i < numConstraints; i++) {
		genericFree(constraints[i]);
	}
	genericFree(constraints);
	genericFree(sizesAmount);

	for (int i = 0; i < maxConstraintSize + 1; i++) {
		genericFree(constraintsOfSize[i]);
	}
	genericFree(constraintsOfSize);

	cudaDeviceReset();
	return 0;
}