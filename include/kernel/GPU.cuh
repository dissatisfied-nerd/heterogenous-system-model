#pragma once
#include <vector>

class CudaKernel 
{
public:
    static void matrixMul(const std::vector<float> &a, 
                          const std::vector<float> &b, 
                          float *result, 
                          size_t size);

};