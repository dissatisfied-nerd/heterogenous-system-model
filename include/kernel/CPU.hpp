#pragma once
#include <vector>

class CpuKernel 
{
public:    
    static void matrixMul(const std::vector<float>& a, 
                         const std::vector<float>& b,
                         std::vector<float>& result,
                         size_t size);
};