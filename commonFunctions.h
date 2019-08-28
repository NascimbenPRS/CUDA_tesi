// allocate memory using "cudaMallocManaged" if compiled with NVCC, "malloc" else
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

// free memory, using "cudaFree" if compiled with NVCC, "free" else
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
