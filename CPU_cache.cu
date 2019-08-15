#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>


// sum array of integers sequentially (using cache)
void arraySum(int *arr, int arraySize, int *sumValue) {
	int tempSum = 0;
	for (int i = 0; i < arraySize; i++) {
		tempSum += arr[i];
	}
	*sumValue = tempSum;
}

int main()
{
	int arraySize = 1 << 21; // 2M integers
	int numCycles = 4000; // # of repetitions
	int *arr;
	cudaMallocManaged(&arr, arraySize * sizeof(int)); // allocate arraySize * 4 bytes
	printf("Sum array of integers on CPU, using cache.\nArray size=  %d integers\n", arraySize);

	// initialize array
	for (int i = 0; i < arraySize; i++) {
		arr[i] = 1;
	}
	int sumValue = 0;
	
	// Time measurement
	int elapsedClocks = 0, startClock = 0, endClock = 0;
	double elapsedTime, avgElapsedTime;

	startClock = clock();
	for (int j = 0; j < numCycles; j++) {
		arraySum(arr, arraySize, &sumValue);

	}
	endClock = clock();

	// Print results
	printf("startClock: %d\n", startClock);
	printf("endClock: %d\n", endClock);
	elapsedClocks = endClock - startClock;
	printf("elapsedClock: %d\n", elapsedClocks);
	elapsedTime = ((double)(elapsedClocks)) / (CLOCKS_PER_SEC);
	avgElapsedTime = elapsedTime / numCycles;
	printf("Sum= %d. Number of repetitions= %d.\nElapsed time= %fs. Average elapsed time= %fs.\n", sumValue, numCycles, elapsedTime, avgElapsedTime);

	cudaFree(arr);

	return 0;
}
