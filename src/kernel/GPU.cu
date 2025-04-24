#include "kernel/GPU.cuh"
#include <chrono>

__global__ void matrixMulKernel(const double* A, const double* B, double* C, int aRows, int aCols, int bCols) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < aRows && col < bCols) 
    {
        double sum = 0.0;
        
        for (int k = 0; k < aCols; ++k){
            sum += A[row * aCols + k] * B[k * bCols + col];
        }
        
        C[row * bCols + col] = sum;
    }
}

double multiplyGPU(const Task& task) 
{
    int aRows = task.a.size();
    int aCols = task.a[0].size();
    int bCols = task.b[0].size();

    std::vector<double> A(aRows * aCols), B(task.b.size() * bCols), C(aRows * bCols);

    for (int i = 0; i < aRows; ++i){
        for (int j = 0; j < aCols; ++j){
            A[i * aCols + j] = task.a[i][j];
        }
    }

    for (int i = 0; i < (int)task.b.size(); ++i)
        for (int j = 0; j < bCols; ++j)
            B[i * bCols + j] = task.b[i][j];

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, A.size() * sizeof(double));
    cudaMalloc(&d_B, B.size() * sizeof(double));
    cudaMalloc(&d_C, C.size() * sizeof(double));

    cudaMemcpy(d_A, A.data(), A.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    dim3 blockDim(16, 16);
    dim3 gridDim((bCols + 15) / 16, (aRows + 15) / 16);
    matrixMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, aRows, aCols, bCols);

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(C.data(), d_C, C.size() * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return std::chrono::duration<double, std::milli>(end - start).count();
}
