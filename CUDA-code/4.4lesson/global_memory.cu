/*********************************************************************************************
 * file name  :global_memory.cu
 * author     : 权 双
 * date       : 2023-12-26
 * brief      : 静态全局变量使用
***********************************************************************************************/

#include <cuda_runtime.h>
#include <iostream>
#include "common.cuh"

__device__ int d_x = 1;
__device__ int d_y[2];

__global__ void kernel(void)
{
    d_y[0] += d_x;
    d_y[1] += d_x;

    printf("d_x = %d, d_y[0] = %d, d_y[1] = %d.\n", d_x, d_y[0], d_y[1]);
}



int main(int argc, char **argv)
{
    int devID = 0;
    cudaDeviceProp deviceProps;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProps, devID));
    std::cout << "运行GPU设备:" << deviceProps.name << std::endl;

    int h_y[2] = {10, 20};
    CUDA_CHECK(cudaMemcpyToSymbol(d_y, h_y, sizeof(int) * 2));

    dim3 block(1);
    dim3 grid(1);
    kernel<<<grid, block>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpyFromSymbol(h_y, d_y, sizeof(int) * 2));
    printf("h_y[0] = %d, h_y[1] = %d.\n", h_y[0], h_y[1]);

    CUDA_CHECK(cudaDeviceReset());

    return 0;
}