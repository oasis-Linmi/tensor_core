
#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cstdlib>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cout << "No CUDA-capable devices found." << std::endl;
        return 1;
    }

    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        std::cout << "Device " << i << " - " << deviceProp.name << std::endl;
        std::cout << "   Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "   Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "   Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "   Clock Rate: " << deviceProp.clockRate / 1000 << " MHz" << std::endl;
        std::cout << "   Multi-Processor Count: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "--------------------------------------------------------" << std::endl;
    }

    return 0;
}
