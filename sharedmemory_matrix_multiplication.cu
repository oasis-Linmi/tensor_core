#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define N 32
#define BLOCK_SIZE 16
__global__ void mul_gpu(const int *a, const int *b, int *c)
{
    __shared__ int s_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int s_b[BLOCK_SIZE][BLOCK_SIZE];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;
    int sum = 0;
    for (int k = 0; k < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; k++)
    {
        if (row < N && k * BLOCK_SIZE + tx < N)
        {
            s_a[ty][tx] = a[row * N + k * BLOCK_SIZE + tx];
        }
        else
        {
            s_a[ty][tx] = 0;
        }
        if (col < N && k * BLOCK_SIZE + ty < N)
        {
            s_b[ty][tx] = b[(k * BLOCK_SIZE + ty) * N + col];
        }
        else
        {
            s_b[ty][tx] = 0;
        }
        __syncthreads();
        for (int i = 0; i < BLOCK_SIZE; i++)
        {
            sum += s_a[ty][i] * s_b[i][tx];
        }
        __syncthreads();
    }
    if (row < N && col < N)
    {
        c[row * N + col] = sum;
    }
}
int main()
{
    cudaEvent_t start, stop;
    float gpu_time;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Wrap Size: %d\n", deviceProp.warpSize);
    printf("Start Executing...\n");
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    int size = N * N;
    int *a, *b, *c;
    int *vec1, *vec2, *vec3;
    a = (int *)malloc(size * sizeof(int));
    b = (int *)malloc(size * sizeof(int));
    c = (int *)malloc(size * sizeof(int));
    cudaMalloc((void **)&vec1, size * sizeof(int));
    cudaMalloc((void **)&vec2, size * sizeof(int));
    cudaMalloc((void **)&vec3, size * sizeof(int));
    for (int i = 0; i < size; i++)
    {
        a[i] = 1;
        b[i] = 1;
    }
    cudaMemcpy(vec1, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(vec2, b, size * sizeof(int), cudaMemcpyHostToDevice);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    mul_gpu<<<grid, block>>>(vec1, vec2, vec3);
    cudaMemcpy(c, vec3, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(vec1);
    cudaFree(vec2);
    cudaFree(vec3);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    for (int i = 0; i < size; i++)
    {
        if (i % N == 0)
            printf("\n");
        printf("%d ", c[i]);
    }
    printf("\nTotal time: %fms\n", gpu_time);
    free(a);
    free(b);
    free(c);
    return 0;
}
