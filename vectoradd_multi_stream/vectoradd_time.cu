/*--------------------------- vectoradd_time.cu -------------------------------------
 |  File vectoradd.cu
 |
 |  Description:  kernel function: vectorAdd 
 |                input: A,B
 |                output: C
 |                
 |                一个stream执行10次kernel并记录每次执行时间
 |                使用cudaDeviceReset()重置设备
 |                将d_A d_B设置为持久数据
 |  Version: 1.0
 *-----------------------------------------------------------------------*/
#include <cuda_runtime.h>
#include <stdio.h>
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

// 设置持久缓存访问属性
void setPersistentCache(void* base_ptr, size_t num_bytes, cudaStream_t& stream) {
    cudaStreamAttrValue stream_attribute;
    memset(&stream_attribute, 0, sizeof(stream_attribute));
    stream_attribute.accessPolicyWindow.base_ptr = base_ptr;
    stream_attribute.accessPolicyWindow.num_bytes = num_bytes;
    stream_attribute.accessPolicyWindow.hitRatio = 1.0; // 假设高命中率
    stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

    // 设置流属性
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
}


int main() {
    int numElements = 1024;
    size_t size = numElements * sizeof(float);
    int numTrials = 10;

    // 分配主机内存和设备内存
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    cudaMallocHost(&h_A, size);
    cudaMallocHost(&h_B, size);
    cudaMallocHost(&h_C, size);
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 初始化数组
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // 创建 CUDA 流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 设置持久缓存访问属性
    //setPersistentCache(d_A, size, stream);
    //setPersistentCache(d_B, size, stream);

    // 执行核函数并记录执行时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds[numTrials];

    for (int i = 0; i < numTrials; ++i) {
        // 拷贝数据到设备
        cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream);

        // 记录开始时间
        cudaEventRecord(start, stream);

        // 执行核函数
        vectorAdd<<<(numElements + 255) / 256, 256, 0, stream>>>(d_A, d_B, d_C, numElements);

        // 记录结束时间并计算时间差
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float time;
        cudaEventElapsedTime(&time, start, stop);
        milliseconds[i] = time;

        // 拷贝结果回主机
        cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream);
    }

    // 打印每次执行时间
    for (int i = 0; i < numTrials; ++i) {
        std::cout << "Trial " << i + 1 << " time: " << milliseconds[i] << " ms" << std::endl;
    }

    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 重置设备
    cudaError_t err = cudaDeviceReset();
    if (err != cudaSuccess) {
        std::cerr << "Failed to reset the device: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    return 0;
}
