#pragma once
#include <cuda_runtime.h>

class GPUMemoryPool 
{
private:
    cudaMemPool_t memPool;
public:
    GPUMemoryPool();
    ~GPUMemoryPool();

    void* mallocAsync(size_t size, cudaStream_t stream = 0);
    void freeAsync(void* ptr, cudaStream_t stream = 0);
};
