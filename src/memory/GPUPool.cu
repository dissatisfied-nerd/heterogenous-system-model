#include "memory/GPUPool.cuh"
#include <stdexcept>

GPUMemoryPool::GPUMemoryPool() {
    cudaDeviceGetDefaultMemPool(&memPool, 0);
}

GPUMemoryPool::~GPUMemoryPool() {}

void* GPUMemoryPool::mallocAsync(size_t size, cudaStream_t stream) 
{
    void* ptr = nullptr;
    cudaMallocAsync(&ptr, size, stream);

    if (!ptr) {
        throw std::runtime_error("Failed to cudaMallocAsync");
    }

    return ptr;
}

void GPUMemoryPool::freeAsync(void* ptr, cudaStream_t stream) {
    cudaFreeAsync(ptr, stream);
}
