#include "kernel/CPU.hpp"
#include <immintrin.h>
#include <stdexcept>

void CpuKernel::matrixMul(const std::vector<float>& a,
                         const std::vector<float>& b,
                         std::vector<float>& result,
                         size_t size) 
{
    #pragma omp parallel for collapse(2)
    
    for (size_t i = 0; i < size; ++i) 
    {
        for (size_t k = 0; k < size; ++k) 
        {
            float sum = 0.0f;
        
            for (size_t j = 0; j < size; ++j) {
                sum += a[i * size + j] * b[j * size + k];
            }
        
            result[i * size + k] = sum;
        }
    }
}