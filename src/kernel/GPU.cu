#include "kernel/GPU.cuh"
#include <chrono>

__global__ void matrixMulKernel(const double* A, const double* B, double* C,
    int aRows, int aCols, int bCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < aRows && col < bCols) 
    {
        double sum = 0.0;
        for (int k = 0; k < aCols; ++k) {
            sum += A[row * aCols + k] * B[k * bCols + col];
        }
        C[row * bCols + col] = sum;
    }
}

std::pair<double, double> multiplyGPU(const std::vector<std::vector<double>>& a,
                                      const std::vector<std::vector<double>>& b,
                                      GPUMemoryPool& memoryPool) 
{
    int aRows = a.size();
    int aCols = a[0].size();
    int bCols = b[0].size();

    std::vector<double> A(aRows * aCols);
    std::vector<double> B(b.size() * bCols);
    std::vector<double> C(aRows * bCols);

    for (int i = 0; i < aRows; ++i)
        for (int j = 0; j < aCols; ++j)
            A[i * aCols + j] = a[i][j];

    for (int i = 0; i < (int)b.size(); ++i)
        for (int j = 0; j < bCols; ++j)
            B[i * bCols + j] = b[i][j];

    // Используем пул памяти
    double* d_A = static_cast<double*>(memoryPool.mallocAsync(A.size() * sizeof(double)));
    double* d_B = static_cast<double*>(memoryPool.mallocAsync(B.size() * sizeof(double)));
    double* d_C = static_cast<double*>(memoryPool.mallocAsync(C.size() * sizeof(double)));

    // Events
    cudaEvent_t startTransfer, endTransfer, startCompute, endCompute;
    cudaEventCreate(&startTransfer);
    cudaEventCreate(&endTransfer);
    cudaEventCreate(&startCompute);
    cudaEventCreate(&endCompute);

    // Transfer host → device
    cudaEventRecord(startTransfer);
    cudaMemcpy(d_A, A.data(), A.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaEventRecord(endTransfer);

    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((bCols + 15) / 16, (aRows + 15) / 16);
    cudaEventRecord(startCompute);
    matrixMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, aRows, aCols, bCols);
    cudaEventRecord(endCompute);

    // Transfer result device → host
    cudaMemcpy(C.data(), d_C, C.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Measure timings
    float transferTime = 0.0f, computeTime = 0.0f;
    cudaEventElapsedTime(&transferTime, startTransfer, endTransfer);
    cudaEventElapsedTime(&computeTime, startCompute, endCompute);

    // Освобождаем память через пул
    memoryPool.freeAsync(d_A);
    memoryPool.freeAsync(d_B);
    memoryPool.freeAsync(d_C);

    cudaEventDestroy(startTransfer);
    cudaEventDestroy(endTransfer);
    cudaEventDestroy(startCompute);
    cudaEventDestroy(endCompute);

    return {computeTime, transferTime}; // в миллисекундах
}

