#include <iostream>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h> // 使用 half 需要包含此头文件

using namespace nvcuda; // WMMA API 所在的命名空间

// 矩阵大小
#define MATRIX_M 16
#define MATRIX_N 16
#define MATRIX_K 16

// CUDA 核函数：使用 Tensor Core 进行矩阵乘法
__global__ void matrixMultiplyTensorCore(const half *a, const half *b, half *c, int M, int N, int K)
{
    // 使用 WMMA 声明 fragment
    wmma::fragment<wmma::matrix_a, MATRIX_M, MATRIX_N, MATRIX_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, MATRIX_M, MATRIX_N, MATRIX_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, MATRIX_M, MATRIX_N, MATRIX_K, half> c_frag;

    // 初始化 accumulator
    wmma::fill_fragment(c_frag, 0.0f);

    // 将数据加载到 fragment 中
    wmma::load_matrix_sync(a_frag, a, K);
    wmma::load_matrix_sync(b_frag, b, N);

    // 进行矩阵乘法
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // 将计算结果存储到输出矩阵
    wmma::store_matrix_sync(c, c_frag, N, wmma::mem_row_major);
}

int main()
{
    // 分配主机内存并初始化
    half *h_A = new half[MATRIX_M * MATRIX_K];
    half *h_B = new half[MATRIX_K * MATRIX_N];
    half *h_C = new half[MATRIX_M * MATRIX_N];

    for (int i = 0; i < MATRIX_M * MATRIX_K; i++)
        h_A[i] = __float2half(1.0f); // 初始化为 1.0
    for (int i = 0; i < MATRIX_K * MATRIX_N; i++)
        h_B[i] = __float2half(1.0f); // 初始化为 1.0

    // 分配设备内存
    half *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, MATRIX_M * MATRIX_K * sizeof(half));
    cudaMalloc((void **)&d_B, MATRIX_K * MATRIX_N * sizeof(half));
    cudaMalloc((void **)&d_C, MATRIX_M * MATRIX_N * sizeof(half));

    // 将数据从主机复制到设备
    cudaMemcpy(d_A, h_A, MATRIX_M * MATRIX_K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, MATRIX_K * MATRIX_N * sizeof(half), cudaMemcpyHostToDevice);

    // 启动 CUDA 核函数
    dim3 grid(1);
    dim3 block(32, 1, 1); // 每个线程块 32 个线程
    matrixMultiplyTensorCore<<<grid, block>>>(d_A, d_B, d_C, MATRIX_M, MATRIX_N, MATRIX_K);

    // 将结果从设备复制回主机
    cudaMemcpy(h_C, d_C, MATRIX_M * MATRIX_N * sizeof(half), cudaMemcpyDeviceToHost);

    // 输出结果
    std::cout << "Result Matrix C:" << std::endl;
    for (int i = 0; i < MATRIX_M; i++)
    {
        for (int j = 0; j < MATRIX_N; j++)
        {
            std::cout << __half2float(h_C[i * MATRIX_N + j]) << " ";
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

