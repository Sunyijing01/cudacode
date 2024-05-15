/*--------------------------- vectoradd_l2.cu -------------------------------------
 |  File vectoradd_l2.cu
 |
 |  Description:  在vectoradd.cu的基础上加上设置持久缓存
 |  Version: 1.0
 *-----------------------------------------------------------------------*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <iostream>

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
  }
}

int main() {
    int N = 1024;  // 向量大小
    size_t size = N * sizeof(float);
    
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
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 拷贝数据到设备内存
    cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream);

    // 设置流属性
    cudaStreamAttrValue stream_attribute;
    stream_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(d_A);
    stream_attribute.accessPolicyWindow.num_bytes = size;
    stream_attribute.accessPolicyWindow.hitRatio = 0.6;
    stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);


    // 配置执行参数
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 在每个流上多次启动核函数
    for (int i = 0; i < 10; ++i) {
        vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, N);
    }

    // 拷贝结果回主机内存
    cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream);

    // 同步流
    cudaStreamSynchronize(stream);

    // 清理资源
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaStreamDestroy(stream);

    // 重置设备
    cudaError_t err = cudaDeviceReset();
    if (err != cudaSuccess) {
        std::cerr << "Failed to reset the device: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    return 0;
}
