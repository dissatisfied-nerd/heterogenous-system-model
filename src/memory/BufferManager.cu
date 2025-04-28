#include "memory/BufferManager.cuh"

BufferManager::BufferManager(GPUMemoryPool& poolRef) : pool(poolRef) {}

BufferManager::Buffers& BufferManager::getBuffers(size_t aRows, size_t aCols, size_t bCols) 
{
    std::string key = std::to_string(aRows) + "x" + std::to_string(aCols) + "x" + std::to_string(bCols);
    if (buffers.find(key) == buffers.end()) 
    {
        Buffers buf;
        buf.aSize = aRows * aCols;
        buf.bSize = aCols * bCols;
        buf.cSize = aRows * bCols;
        buf.d_A = static_cast<double*>(pool.mallocAsync(buf.aSize * sizeof(double)));
        buf.d_B = static_cast<double*>(pool.mallocAsync(buf.bSize * sizeof(double)));
        buf.d_C = static_cast<double*>(pool.mallocAsync(buf.cSize * sizeof(double)));
        buffers[key] = buf;
    }

    return buffers[key];
}

BufferManager::~BufferManager() 
{
    for (auto& [_, buf] : buffers) 
    {
        pool.freeAsync(buf.d_A);
        pool.freeAsync(buf.d_B);
        pool.freeAsync(buf.d_C);
    }
}
