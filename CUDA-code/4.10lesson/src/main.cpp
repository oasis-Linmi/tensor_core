#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <string>
#include "utils.hpp"
#include "timer.hpp"
#include "reduce.hpp"
#include <cstring>
#include <memory>
#include <cmath>

int seed;
int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "用法: ./build/reduction [size] [blockSize]" << std::endl;
        return -1;
    }

    Timer timer;
    char str[100];
    int size = std::stoi(argv[1]);
    int blockSize = std::stoi(argv[2]);
   
    int gridsize = size / blockSize;
    
    float* h_idata = nullptr;
    float* h_odata = nullptr;
    h_idata = (float*)malloc(size * sizeof(float));
    h_odata = (float*)malloc(gridsize * sizeof(float));

    seed = 1;
    initMatrix(h_idata, size, seed);
    memset(h_odata, 0, gridsize * sizeof(float));

    // CPU归约
    timer.start_cpu();
    float sumOnCPU = ReduceOnCPU(h_idata, size);
    timer.stop_cpu();
    std::sprintf(str, "reduce in cpu, result:%f", sumOnCPU);
    timer.duration_cpu<Timer::ms>(str);

    // GPU warmup 
    timer.start_gpu();
    ReduceOnGPUWithDivergence(h_idata, h_odata, size, blockSize);
    timer.stop_gpu();
    // timer.duration_gpu("reduce in gpu(warmup)");


    // GPU归约(带分支)
    timer.start_gpu();
    ReduceOnGPUWithDivergence(h_idata, h_odata, size, blockSize);
    timer.stop_gpu();
    float sumOnGPUWithDivergence = 0;
    for (int i = 0; i < gridsize; i++) sumOnGPUWithDivergence += h_odata[i];
    std::sprintf(str, "reduce in gpu with divergence, result:%f", sumOnGPUWithDivergence);
    timer.duration_gpu(str);

    // GPU归约(不带分支)
    timer.start_gpu();
    ReduceOnGPUWithoutDivergence(h_idata, h_odata, size, blockSize);
    timer.stop_gpu();
    float sumOnGPUWithoutDivergence = 0;
    for (int i = 0; i < gridsize; i++) sumOnGPUWithoutDivergence += h_odata[i];
    std::sprintf(str, "reduce in gpu without divergence, result:%f", sumOnGPUWithoutDivergence);
    timer.duration_gpu(str);
    
    free(h_idata);
    free(h_odata);
    return 0;
}
