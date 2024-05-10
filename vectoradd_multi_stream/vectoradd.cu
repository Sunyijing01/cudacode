/*--------------------------- vectoradd.cu -------------------------------------
 |  File vectoradd.cu
 |
 |  Description:  kernel function: vectorAdd 
 |                input: A,B
 |                output: C
 |                可获取当前线程的smid
 |                两个stream各执行10次kernel function
 |                使用cudaDeviceReset()重置设备
 |  Version: 1.0
 *-----------------------------------------------------------------------*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <iostream>

// 使用内联汇编获取当前线程的 SM ID
__device__ __inline__ uint32_t getSMID() {
    uint32_t smid;
    asm volatile ("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

// 核函数定义
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  float p = 100;

  //尽量使kernel执行时间长一些
  if (i < N) {
    for (int j = 0; j < 100000; j++){
      p = p / 2.0;
    }
    for (int j = 0; j < 100000; j++){
      p = p * 2.0;
    }
    for (int j = 0; j < 100000; j++){
      p = p / 1.5;
    }
    for (int j = 0; j < 100000; j++){
      p = p * 1.5;
    }
    C[i] = A[i] + B[i] + p;

    //uint32_t smid = getSMID();
    //printf("SM ID: %u, Block ID: %d, Thread ID: %d\n", smid, blockIdx.x, threadIdx.x);
  }
}

int main() {
    int N = 1024;  // 向量大小
    size_t size = N * sizeof(float);
    // sizeof(float) = 4
    
    float *h_A, *h_B, *h_C; // 主机内存指针
    float *d_A, *d_B, *d_C; // 设备内存指针

    // 分配主机内存
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // 初始化数组
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // 在GPU上分配内存
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 创建CUDA流
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // 拷贝数据到设备内存
    cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream2);

    // 配置执行参数
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 在每个流上多次启动核函数
    for (int i = 0; i < 10; ++i) {
        vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_A, d_B, d_C, N);
        vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_A, d_B, d_C, N);
    }

    // 拷贝结果回主机内存
    cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream2);

    // 同步流
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // 清理资源
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    // 重置设备
    cudaError_t err = cudaDeviceReset();
    if (err != cudaSuccess) {
        std::cerr << "Failed to reset the device: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    return 0;
}
