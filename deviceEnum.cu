/*----------------------------- deviceEnum.cu -----------------------------
|  File deviceEnum.cu
|
|  Description: 查看device数量及其compute capability
|
|  Version: 1.0
*-----------------------------------------------------------------------*/
#include <iostream>
int main(){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device;
    for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %d has compute capability %d.%d.\n",
            device, deviceProp.major, deviceProp.minor);
    }
}
