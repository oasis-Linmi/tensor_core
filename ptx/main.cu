#include <iostream>
#include <vector>
#include <cuda.h>

int main()
{
    const int N = 1024;              // Array size
    std::vector<float> h_A(N, 1.0f); // Host array A
    std::vector<float> h_B(N, 2.0f); // Host array B
    std::vector<float> h_C(N, 0.0f); // Host array C

    float *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_A, h_A.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Load PTX module
    CUmodule module;
    
    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS)
    {
        const char *errMsg;
        cuGetErrorString(err, &errMsg);
        std::cerr << "CUDA Initialization Error: " << errMsg << std::endl;
        return EXIT_FAILURE;
    }

    if (cuModuleLoad(&module, "/home/huanqi/projects/cuda_study/ptx/add.ptx") != CUDA_SUCCESS)
    {
        std::cerr << "Failed to load PTX module" << std::endl;
        return EXIT_FAILURE;
    }

    // Get kernel function
    CUfunction vector_add;
    cuModuleGetFunction(&vector_add, module, "vector_add");

    // Set up kernel arguments
    void *args[] = {&d_A, &d_B, &d_C, (void*)&N};

    // Launch kernel
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    cuLaunchKernel(vector_add,
                   blocks_per_grid, 1, 1,
                   threads_per_block, 1, 1,
                   0, 0, args, 0);

    // Copy result back to host
    cudaMemcpy(h_C.data(), d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    cuModuleUnload(module);
    for (int i = 0; i < N; ++i)
    {
        std::cout << h_C[i] << " ";
    }
    // Verify result
    for (int i = 0; i < N; ++i)
    {
        if (h_C[i] != h_A[i] + h_B[i])
        {
            std::cerr << "Verification failed at index " << i << "\n";
            return EXIT_FAILURE;
        }
    }

    std::cout << "Verification passed!\n";
    return EXIT_SUCCESS;
}