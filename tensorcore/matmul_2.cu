// Author: GPT Have bugs!
#include <iostream>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h> // 使用 half 需要包含此头文件

using namespace nvcuda; // WMMA API 所在的命名空间

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

// CUDA 核函数：使用 Tensor Core 进行矩阵乘法，支持任意大小矩阵
__global__ void matrixMultiplyTensorCore(const half *a, const half *b, half *c, int M, int N, int K)
{
    // 计算每个线程块处理的起始坐标
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) * TILE_M;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) * TILE_N;

    // 定义 WMMA fragments
    wmma::fragment<wmma::matrix_a, TILE_M, TILE_N, TILE_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE_M, TILE_N, TILE_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE_M, TILE_N, TILE_K, half> c_frag;

    // 初始化 accumulator
    wmma::fill_fragment(c_frag, 0.0f);

    // 累加计算 tile
    for (int k = 0; k < K; k += TILE_K)
    {
        // 检查边界并加载矩阵 A 和 B 的片段到 fragment
        if (warpM < M && k < K)
        {
            wmma::load_matrix_sync(a_frag, a + warpM * K + k, K);
        }
        if (warpN < N && k < K)
        {
            wmma::load_matrix_sync(b_frag, b + k * N + warpN, N);
        }

        // 执行 Tensor Core 的矩阵乘法累加
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 将计算结果存储回矩阵 C
    if (warpM < M && warpN < N)
    {
        wmma::store_matrix_sync(c + warpM * N + warpN, c_frag, N, wmma::mem_row_major);
    }
}

int main()
{
    // 设置任意大小的矩阵维度
    int M = 32; // 可变行数
    int N = 32; // 可变列数
    int K = 32; // 可变中间维度

    // 分配主机内存并初始化
    half *h_A = new half[M * K];
    half *h_B = new half[K * N];
    half *h_C = new half[M * N];

    for (int i = 0; i < M * K; i++)
        h_A[i] = __float2half(1.0f); // 初始化为 1.0
    for (int i = 0; i < K * N; i++)
        h_B[i] = __float2half(1.0f); // 初始化为 1.0

    // 分配设备内存
    half *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, M * K * sizeof(half));
    cudaMalloc((void **)&d_B, K * N * sizeof(half));
    cudaMalloc((void **)&d_C, M * N * sizeof(half));

    // 将数据从主机复制到设备
    cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);

    // 计算网格和线程块的维度
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    dim3 block(32, 1, 1); // 每个线程块 32 个线程

    // 启动 CUDA 核函数
    matrixMultiplyTensorCore<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    // 将结果从设备复制回主机
    cudaMemcpy(h_C, d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);

    // 输出结果
    std::cout << "Result Matrix C:" << std::endl;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << __half2float(h_C[i * N + j]) << " ";
        }
        std::cout << std::endl;
    }

    // 释放内存
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
