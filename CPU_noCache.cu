#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>



// sum array of integers non-sequentially (not using cache)
void arraySumStride(int *arr, int arraySize, int cacheLineSize, int *sumValue) {
	int tempSum = 0;
	for (int i = 0; i < cacheLineSize; i++) {
		for (int j = i; j < arraySize; j += cacheLineSize) {
			tempSum += arr[j];
		}
	}
	*sumValue = tempSum;
}

int main()
{
	int arraySize = 1 << 22; // 4M integers
	int cacheLineSize = 64 / sizeof(int); // # integers per cache line
	int numCycles = 2000; // # of repetitions
	int *arr;
	cudaMallocManaged(&arr, arraySize * sizeof(int));
	printf("Sum array of integers on CPU, not using cache.\nArray size=  %d integers\n", arraySize);

	// initialize array
	for (int i = 0; i < arraySize; i++) {
		arr[i] = 1;
	}

	
	// Time measurement
	int sumValue = 0;
	int elapsedClocks = 0, startClock = 0, endClock = 0;
	double elapsedTime;

	startClock = clock();
	for (int j = 0; j < numCycles; j++) {
		arraySumStride(arr, arraySize, cacheLineSize, &sumValue);

	}

	endClock = clock();

	// Print results
	printf("startClock: %d\n", startClock);
	printf("endClock: %d\n", endClock);
	elapsedClocks = endClock - startClock;
	printf("elapsedClock: %d\n", elapsedClocks);
	elapsedTime = ((double)(elapsedClocks)) / (CLOCKS_PER_SEC * numCycles);
	printf("Sum = %d, elapsed time= %f s.\n", sumValue, elapsedTime);

	cudaFree(arr);
	return 0;
}
