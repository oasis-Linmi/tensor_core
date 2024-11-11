#include "cuda_runtime.h"
#include "cuda.h"
#include "stdio.h"
#include "utils.hpp"

__global__ void ReduceNeighboredWithDivergence(float *d_idata, float *d_odata, int size){
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    float *idata = d_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (idx >= size) return;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) d_odata[blockIdx.x] = idata[0];
}

__global__ void ReduceNeighboredWithoutDivergence(float *d_idata, float *d_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    float *idata = d_idata + blockIdx.x * blockDim.x;

    // boundary check
    if(idx >= n) return;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        // convert tid into local array index
        int index = 2 * stride * tid;

        if (index < blockDim.x)
        {
            idata[index] += idata[index + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) d_odata[blockIdx.x] = idata[0];
}


void ReduceOnGPUWithDivergence(float *h_idata, float *h_odata, int size, int blockSize)
{
    int ibytes = size * sizeof(float);
    int obytes = size / blockSize * sizeof(float);

    memset(h_odata, 0, obytes);

    float* d_idata = nullptr;
    float* d_odata = nullptr;

    CUDA_CHECK(cudaMalloc(&d_idata, ibytes));
    CUDA_CHECK(cudaMalloc(&d_odata, obytes));

    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, ibytes, cudaMemcpyHostToDevice));
   
    dim3 block(blockSize);
    dim3 grid(size / blockSize);
    ReduceNeighboredWithDivergence <<<grid, block>>> (d_idata, d_odata, size);

    // 将结果从device拷贝回host
    CUDA_CHECK(cudaMemcpy(h_odata, d_odata, obytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    //注意在同步后，检测核函数
    CUDA_KERNEL_CHECK();  

    CUDA_CHECK(cudaFree(d_odata));
    CUDA_CHECK(cudaFree(d_idata));
}

void ReduceOnGPUWithoutDivergence(float *h_idata, float *h_odata, int size, int blockSize)
{
    int ibytes = size * sizeof(float);
    int obytes = size / blockSize * sizeof(float);

    memset(h_odata, 0, obytes);
    
    float* d_idata = nullptr;
    float* d_odata = nullptr;

    CUDA_CHECK(cudaMalloc(&d_idata, ibytes));
    CUDA_CHECK(cudaMalloc(&d_odata, obytes));

    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, ibytes, cudaMemcpyHostToDevice));
    
    dim3 block(blockSize);
    dim3 grid(size / blockSize);
    ReduceNeighboredWithoutDivergence <<<grid, block>>> (d_idata, d_odata, size);

    CUDA_CHECK(cudaMemcpy(h_odata, d_odata, obytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_KERNEL_CHECK();

    CUDA_CHECK(cudaFree(d_odata));
    CUDA_CHECK(cudaFree(d_idata));
}

