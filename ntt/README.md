在 CUDA 中，原生的 128 位整数（如 __int128 或 unsigned __int128）不受支持。因此，编译器会报错。要在 CUDA 设备代码中实现 128 位整数运算，我们需要手动管理 128 位整数，通常可以将其表示为两个 64 位整数。下面我们将修改你的代码，以便在设备代码中正确处理 128 位整数。

修改 128 位整数运算
首先，我们定义一个 128 位整数结构体，并实现基本运算，例如乘法和取模。接着，我们将你的 CUDA 核函数转换为使用这个结构体。

解释更改
定义 uint128_t 结构：用于模拟 128 位整数。包含两个 64 位整数（low 和 high）。
定义乘法函数 mul128：用于计算两个 64 位整数的乘积，结果保存在 uint128_t 结构中。
定义取模函数 mod128：用于计算 128 位整数对 64 位整数取模的结果。
修改 d_qmul 和 d_qpow 使用 uint128_t：确保这些函数使用新定义的 128 位整数运算函数。
修改内核函数 ntt_kernel：确保内核函数按需使用新的 128 位整数运算。
通过这些更改，可以避免在 CUDA 设备代码中直接使用不受支持的 128 位整数，同时仍能在设备中实现必要的高精度运算。
