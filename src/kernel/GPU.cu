#include "kernel/GPU.cuh"
#include <stdexcept>

__global__ void matrixMulKernel(const float* a, const float* b, float* result, size_t size) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < size && col < size) 
    {
        float sum = 0.0f;
    
        for (size_t k = 0; k < size; ++k) {
            sum += a[row * size + k] * b[k * size + col];
        }
    
        result[row * size + col] = sum;
    }
}

void CudaKernel::matrixMul(const std::vector<float> &a, 
                           const std::vector<float> &b, 
                           float *result, 
                           size_t size)
{
    float *d_a, *d_b, *d_result;
    size_t bytes = size * size * sizeof(float);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_result, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                      (size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, size);

    cudaMemcpy(result, d_result, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}
