/*--------------------------- main.cu -------------------------------------
 |  File main.cu
 |
 |  Description:  
 |
 |  Version: 1.0
 *-----------------------------------------------------------------------*/

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

#include <cuda_runtime.h>

// CUDA helper functions
#include <helper_functions.h>
#include <helper_cuda.h>

// GPU benchmarks
#include "benchmarks.cuh"


//#define PRINT_ALL


int main(){
    // Retrieve device properties
    cudaDeviceProp device_prop;
    int current_device = 0;
    checkCudaErrors(cudaGetDevice(&current_device));
    checkCudaErrors(cudaGetDeviceProperties(&device_prop, current_device));
    unsigned persisting_l2_cache_max_size_mb = device_prop.persistingL2CacheMaxSize / 1024 / 1024;
    
    #ifdef PRINT_ALL
   	printf("\n***** Basic Info *****\n");
    	printf("GPU: %s \n", device_prop.name);
    	printf("L2 Cache Size: %u MB \n", device_prop.l2CacheSize / 1024 / 1024);
    	printf("Max Persistent L2 Cache Size: %u MB \n\n", persisting_l2_cache_max_size_mb);
        printf("multiProcessorCount : %d\n", device_prop.multiProcessorCount);
        printf("maxThreadsPerMultiProcessor : %d\n", device_prop.maxThreadsPerMultiProcessor);
    #endif


    int numElements = 500;
    size_t size = numElements * sizeof(float);

    cudaStream_t stream1, stream2;
    checkCudaErrors(cudaStreamCreate(&stream1));
    checkCudaErrors(cudaStreamCreate(&stream2));

    cudaEvent_t event_a;
    cudaEventCreate(&event_a);

    // Allocate the host input vectors
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    // Allocate the host output vector
    float *h_C = (float *)malloc(size);
    float *h_D = (float *)malloc(size);

    // Initialize the host input vectors
    for (int i = 0; i < numElements; i++){
        h_A[i] = 1;
        h_B[i] = 2;
    }

    // Allocate the device input vectors
    float *d_A;
    float *d_B;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    // Allocate the device output vector
    float *d_C;
    float *d_D;
    cudaMalloc((void **)&d_C, size);
    cudaMalloc((void **)&d_D, size);

    // Copy the host input vectors A and B in host memory to the device input
    cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream2);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    cudaEventRecord(event_a, stream1);
    cudaStreamWaitEvent(stream2, event_a);
        
/*
    // Start to use persistent cache.
    cudaStream_t stream_persistent_cache;
    size_t const num_megabytes_persistent_cache{3};
    checkCudaErrors(cudaStreamCreate(&stream_persistent_cache));

    checkCudaErrors(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, num_megabytes_persistent_cache * 1024 * 1024));

    cudaStreamAttrValue stream_attribute_thrashing;
    stream_attribute_thrashing.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(d_A);
    stream_attribute_thrashing.accessPolicyWindow.num_bytes = size;
    stream_attribute_thrashing.accessPolicyWindow.hitRatio = 1.0;
    stream_attribute_thrashing.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    stream_attribute_thrashing.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

    checkCudaErrors(cudaStreamSetAttribute(stream_persistent_cache, cudaStreamAttributeAccessPolicyWindow, &stream_attribute_thrashing));
*/

    vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_A, d_B, d_C, numElements);
    vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_A, d_B, d_D, numElements);

    // Copy the device result vector in device memory to the host result vector
    cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(h_D, d_D, size, cudaMemcpyDeviceToHost, stream2);

    // Free device global memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);



//--------------------------------------------------------------//
/*
    cudaStream_t stream1, stream2;
    checkCudaErrors(cudaStreamCreate(&stream1));
    checkCudaErrors(cudaStreamCreate(&stream2));


    size_t size_persistent = multiplier*1024*1024/sizeof(int);
    size_t size_streaming = 100*1024*1024/sizeof(int);
    dim3 dimsA(multiplier * SM_MIN_OPT_THREADS_SIZE, multiplier * SM_MIN_OPT_THREADS_SIZE / 2, 1);
    dim3 dimsB(3, 3, 1);

    #ifdef PRINT_ALL
        printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);
    #endif

    #ifdef PRINT_ALL
        printf("Vectors size (%d,%d)\n", size_persistent, size_streaming);
    #endif
    
    launch_reset_data(BLOCK_SIZE, size_streaming, size_persistent, stream1);
    //MatrixConvolution(SM_MIN_OPT_THREADS_SIZE, dimsA, dimsB, stream2);
    //launch_reset_data(BLOCK_SIZE, size_streaming, size_persistent, stream1);


    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
*/
}
