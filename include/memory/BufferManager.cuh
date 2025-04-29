#pragma once

#include <unordered_map>
#include <string>
#include "memory/GPUPool.cuh"

class BufferManager 
{
private:
    GPUMemoryPool& pool;

    struct Buffers 
    {
        double* d_A;
        double* d_B;
        double* d_C;
        size_t aSize;
        size_t bSize;
        size_t cSize;
    };

    std::unordered_map<std::string, Buffers> buffers;

public:
    BufferManager(GPUMemoryPool& poolRef);
    Buffers& getBuffers(size_t aRows, size_t aCols, size_t bCols);
    ~BufferManager();
};
