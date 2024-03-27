/*----------------------------- benchmarks.cuh -----------------------------
|  File benchmarks.cuh
|
|  Description: 
|
|  Version: 1.0
*-----------------------------------------------------------------------*/

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, char const* const func, char const* const file,
           int const line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(char const* const file, int const line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

static __device__ __inline__ uint32_t __mysmid(){
  uint32_t ssmid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(ssmid));
  return ssmid;}
 
static __device__ __inline__ uint32_t __mywarpid(){
  uint32_t warpid;
  asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpid));
  return warpid;}
 
static __device__ __inline__ uint32_t __mylaneid(){
  uint32_t laneid;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid));
  return laneid;}


 /* reset_data
 *
 * Description: CUDA kernel for reseting the values of a vector. 
 *		Obtained from Lei Mao's blog: 
 *		https://leimao.github.io/blog/CUDA-L2-Persistent-Cache/  
 *
 * Parameter:   
 *		- int* data_streaming: Vector to reset
 *		- int const* lut_persistent: Vector used to reset data_streaming with
 *		- size_t data_streaming_size: Size of data_streaming
 *		- size_t lut_persistent_size: Size of lut_persistent
 *
 * Returns:     Nothing
 *
 * */
__global__ void reset_data(int* data_streaming, int const* lut_persistent, size_t data_streaming_size, size_t lut_persistent_size){
    size_t const idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t const stride = blockDim.x * gridDim.x;
   
    for (size_t i = idx; i < data_streaming_size; i += stride)
    	data_streaming[i] = lut_persistent[i % lut_persistent_size];
		
	//printf("I am thread %d, my SM ID is %d, my warp ID is %d, and my warp lane is %d\n", idx, __mysmid(), __mywarpid(), __mylaneid());
}


__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int p = 0;

  if (i < numElements) {
	for (int j = 0; j < 1000000; j++){
		p = p + 1;
	}
	for (int j = 0; j < 1000000; j++){
		p = p - 1;
	}
    C[i] = A[i] + B[i] + p;
  }
}


/* conv2DCUDA
 *
 * Description: CUDA kernel for the 2D convolution of two matrices. 
 *		 Borders are not processed.
 *		 The block ID is used for selecting the input matrix row.
 *
 * template:
 *		- int BLOCK_SIZE: Number of threads making up a block 
 *		- int KERNEL_SIZE: Size of the filter matrix 
 *
 * Parameter:   
 *		- float *imgf: Resultant matrix
 *		- float *img: Input matrix
 *		- float *kernel: Filter matrix
 *		- int Nx: img width
 *		- int Ny: img height
 *
 * Returns:     Nothing
 *
 * */
template <int BLOCK_SIZE, const int KERNEL_SIZE> __global__ void conv2DCUDA(float *imgf, float *img, float *kernel, int Nx, int Ny){

  	// Block index
  	int bx = blockIdx.x;
  
  	// Thread index
	int tx = threadIdx.x;
	
	// the center of the filter
	int center = (KERNEL_SIZE - 1)/2;
	
	// each block is assigned to a row of an image, iy integer index of y
	int iy = bx + center;

	// each thread is assigned to a pixel of a row, ix integer index of x
	int ix = tx + center;

	// Kernel max size
	const int FILTER_MAX_SIZE = KERNEL_SIZE*KERNEL_SIZE;
	
	// Locked data for the current thread block
	__shared__ float sdata[FILTER_MAX_SIZE];

	if (tx<FILTER_MAX_SIZE)
	    sdata[tx] = kernel[tx];

	// Wait until the filter matrix data is locked into the L1 cache
	__syncthreads();
	

	int ii = 0;
	int jj = 0;
	int sum = 0;

	// Across the horizontal 
	for (int cnt = 0; cnt<Nx; cnt+= BLOCK_SIZE){
        	int idx = iy*Nx + (ix + cnt);
        	// Avoid borders
	    	if (idx < Nx*Ny && ix + cnt < Nx-center && ix + cnt != Nx && iy < Ny-center){
	    	    // Apply filter
		    for (int ki = 0; ki<KERNEL_SIZE; ki++){
			for (int kj = 0; kj<KERNEL_SIZE; kj++){
			
			   ii = kj + ix - center + cnt;
			   jj = ki + iy - center;
		    	   sum += img[jj*Nx + ii] * sdata[ki*KERNEL_SIZE + kj];
		    	}	 
		   }
		   
		   // Write element of the final image matrix in the device
	           imgf[idx] = sum;
	           
	           // Reset sum variable
	           sum = 0;
	       }  
	}

}