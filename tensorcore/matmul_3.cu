// Reference: https://zhuanlan.zhihu.com/p/620766588
#include <iostream>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h> // 使用 half 需要包含此头文件

// Function to perform ceiling division
inline __host__ __device__ size_t div_ceil(size_t x, size_t y)
{
    return (x + y - 1) / y;
}

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define WARP_SIZE 32

using namespace nvcuda;

__global__ void wmmaNaiveKernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, size_t M,
                                size_t N, size_t K) {
    const size_t K_tiles = div_ceil(K, WMMA_K);

    const size_t warp_row = blockIdx.y * WMMA_M;
    const size_t warp_col = blockIdx.x * WMMA_N;

    if (warp_row >= M && warp_col >= N) {
        return;
    }

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;

    wmma::fill_fragment(C_frag, 0.0f);

#pragma unroll
    for (size_t i = 0; i < K_tiles; ++i) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> B_frag;

        wmma::load_matrix_sync(A_frag, A + warp_row * K + i * WMMA_K, K);
        wmma::load_matrix_sync(B_frag, B + i * WMMA_K + warp_col * K, K);

        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
    }

    wmma::store_matrix_sync(C + warp_row * N + warp_col, C_frag, N, wmma::mem_row_major);
}

void wmmaNaive(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
    dim3 block(WARP_SIZE);
    dim3 grid(div_ceil(N, WMMA_N), div_ceil(M, WMMA_M));

    wmmaNaiveKernel<<<grid, block>>>(A, B, C, M, N, K);
}

int main()
{
    // 设置任意大小的矩阵维度
    int M = 64; // 可变行数
    int N = 64; // 可变列数
    int K = 63; // 可变中间维度

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

    wmmaNaive(d_A, d_B, d_C, M, N, K);

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