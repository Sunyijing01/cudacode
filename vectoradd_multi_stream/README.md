# cudacode
parallel.cu使用两个stream运行两个kernel

stream1 HtoD vecterADD DtoH

stream2 HtoD vecterADD DtoH


![c324823bbe0559084e2a007cc31b499](https://github.com/Sunyijing01/cudacode/assets/59354764/fa51b196-7542-40f6-8695-7eef79fb8e49)

运行

nvcc -I /usr/local/cuda-11.4/samples/common/inc parallel.cu -o parallel

nsys profile -o output ./parallel
