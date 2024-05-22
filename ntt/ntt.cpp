#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <chrono>
typedef unsigned __int128 uint128_t;//高版本gcc支持128位无符号数，也可以考虑使用gmp库

const int N = 16384;                // NTT 长度
const unsigned long long P = 4179340454199820289; // 大素数 P
const int G = 3;                    // 素数的原根 G

std::vector<int> r(N);


unsigned long long qmul(unsigned long long a, unsigned long long b, unsigned long long mod) {
    __uint128_t res = (__uint128_t)a * b;
    return (unsigned long long)(res % mod);
}

unsigned long long qpow(unsigned long long x, unsigned long long y) // 快速模幂算法
{
    unsigned long long res = 1;
    while (y)
    {
        if (y & 1)
            res = qmul(res,x,P);
        x = qmul(x,x,P);
        y >>= 1;
    }
    return res;
}

void bit_reverse_indices(int n) {  // 计算比特逆序的函数
    int log_n = std::log2(n);
    for (int i = 0; i < n; i++) {
        int x = i, rev = 0;
        for (int j = 0; j < log_n; j++) {
            rev = (rev << 1) | (x & 1);
            x >>= 1;
        }
        r[i] = rev;
    }
}

void ntt(std::vector<unsigned long long>& x, int lim) {
    bit_reverse_indices(lim); // 初始化比特逆序索引
    for (int i = 0; i < lim; ++i) {
        if (r[i] < i) {
            std::swap(x[i], x[r[i]]);
        }
    }
    for (int m = 2; m <= lim; m <<= 1) {
        int k = m >> 1;
        unsigned long long gn = qpow(G, (P - 1) / m);
        for (int i = 0; i < lim; i += m) {
            unsigned long long g = 1;
            for (int j = 0; j < k; j++, g = qmul(g, gn, P)) {
                unsigned long long tmp = qmul(x[i + j + k], g, P);
                x[i + j + k] = (x[i + j] >= tmp ? (x[i + j] - tmp) : (x[i + j] + P - tmp));
                x[i + j] = (x[i + j] + tmp) % P;
            }
        }
    }
}

int main() {
    std::vector<unsigned long long> data(N, 0); // 初始化数据，此处以全0为例，可以根据需要填充
    // 可以在此处对 data 进行赋值操作

    auto start = std::chrono::high_resolution_clock::now();

    ntt(data, N); // 执行 NTT

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "NTT algorithm execution time: " << duration << " microseconds." << std::endl;

    return 0;
}
