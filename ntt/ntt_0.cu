#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <chrono>
typedef __int128 int128_t;

#define N 16384
#define P 4179340454199820289ULL
#define G 3

__device__ unsigned long long d_qmul(unsigned long long a, unsigned long long b, unsigned long long mod) {
    int128_t res = (int128_t)a * b;
    return (unsigned long long)(res % mod);
}

__device__ unsigned long long d_qpow(unsigned long long x, unsigned long long y, unsigned long long p) {
    unsigned long long res = 1;
    while (y) {
        if (y & 1)
            res = d_qmul(res, x, p);
        x = d_qmul(x, x, p);
        y >>= 1;
    }
    return res;
}

__global__ void bit_reverse_indices(int* r, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int log_n = log2f(n);
        int x = idx, rev = 0;
        for (int j = 0; j < log_n; j++) {
            rev = (rev << 1) | (x & 1);
            x >>= 1;
        }
        r[idx] = rev;
    }
}

__global__ void ntt_kernel(unsigned long long* x, int* r, int lim, int m, unsigned long long gn, unsigned long long p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int k = m >> 1;
    int i = (idx / k) * m;
    int j = idx % k;

    if (idx < lim / 2) {
        unsigned long long g = d_qpow(gn, j, p);
        unsigned long long tmp = d_qmul(x[i + j + k], g, p);
        unsigned long long a = x[i + j];
        unsigned long long b = tmp;

        x[i + j + k] = (a >= b ? (a - b) : (a + p - b));
        x[i + j] = (a + b) % p;
    }
}

void ntt(std::vector<unsigned long long>& data, int n) {
    unsigned long long *d_data;
    int *d_r;

    cudaMalloc(&d_data, sizeof(unsigned long long) * n);
    cudaMalloc(&d_r, sizeof(int) * n);
    cudaMemcpy(d_data, data.data(), sizeof(unsigned long long) * n, cudaMemcpyHostToDevice);

    dim3 block_size(256);
    dim3 num_blocks((n + block_size.x - 1) / block_size.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    bit_reverse_indices<<<num_blocks, block_size>>>(d_r, n);
    cudaDeviceSynchronize();

    for (int m = 2; m <= n; m <<= 1) {
        unsigned long long gn = d_qpow(G, (P - 1) / m, P);
        ntt_kernel<<<num_blocks, block_size>>>(d_data, d_r, n, m, gn, P);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(data.data(), d_data, sizeof(unsigned long long) * n, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_r);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "NTT algorithm execution time: " << milliseconds << " milliseconds." << std::endl;
}

int main() {
    std::vector<unsigned long long> data(N, 0);

    ntt(data, N);

    return 0;
}
