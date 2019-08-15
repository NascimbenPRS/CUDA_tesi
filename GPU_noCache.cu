#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <math.h>



// sum array of integers non-sequentially (not using cache)
__global__ void arraySumStride(int *arr, int arraySize, int cacheLineSize, int *sumValue) {
	int tempSum = 0;
	for (int i = 0; i < cacheLineSize; i++) {
		for (int j = i; j < arraySize; j += cacheLineSize) {
			tempSum += arr[j];
		}
	}
	*sumValue = tempSum;
	printf("Valore somma (thread): %d\n", *sumValue);
}

int main()
{
	int arraySize = 1 << 22; // 4M integers
	int cacheLineSize = 128 / sizeof(int); // # integers per cache line
	int numCycles= 10;
	int *arr, *sumValue;
	cudaMallocManaged(&arr, arraySize * sizeof(int));
	cudaMallocManaged(&sumValue, sizeof(int));
	printf("Sum array of integers on GPU (single thread), not using cache.\nArray size=  %d integers\n", arraySize);

	// initialize array
	for (int i = 0; i < arraySize; i++) {
		arr[i] = 1;
	}

	*sumValue = 0;
	

	// Time measurement

	double elapsedTime;
	int elapsedClocks = 0, startClock = 0, endClock = 0;
	startClock = clock();

	arraySumStride << <1, 1 >> > (arr, arraySize, cacheLineSize, sumValue);
	cudaDeviceSynchronize();

	endClock = clock();

	// Print results
	printf("startClock: %d\n", startClock);
	printf("endClock: %d\n", endClock);
	elapsedClocks = endClock - startClock;
	printf("elapsedClock: %d\n", elapsedClocks);
	elapsedTime = ((double)(elapsedClocks)) / CLOCKS_PER_SEC;
	printf("Sum = %d, elapsed time= %f s.\n", *sumValue, elapsedTime);

	cudaFree(arr);
	cudaFree(sumValue);
	return 0;
}