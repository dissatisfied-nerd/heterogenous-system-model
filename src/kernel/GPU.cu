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
    BufferManager& bufferManager) 
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

    // Получаем буферы из менеджера
    auto& buf = bufferManager.getBuffers(aRows, aCols, bCols);

    // Events
    cudaEvent_t startTransfer, endTransfer, startCompute, endCompute;
    cudaEventCreate(&startTransfer);
    cudaEventCreate(&endTransfer);
    cudaEventCreate(&startCompute);
    cudaEventCreate(&endCompute);

    cudaEventRecord(startTransfer);
    cudaMemcpy(buf.d_A, A.data(), buf.aSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(buf.d_B, B.data(), buf.bSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaEventRecord(endTransfer);

    dim3 blockDim(16, 16);
    dim3 gridDim((bCols + 15) / 16, (aRows + 15) / 16);
    cudaEventRecord(startCompute);
    matrixMulKernel<<<gridDim, blockDim>>>(buf.d_A, buf.d_B, buf.d_C, aRows, aCols, bCols);
    cudaEventRecord(endCompute);

    cudaMemcpy(C.data(), buf.d_C, buf.cSize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    float transferTime = 0.0f, computeTime = 0.0f;
    cudaEventElapsedTime(&transferTime, startTransfer, endTransfer);
    cudaEventElapsedTime(&computeTime, startCompute, endCompute);

    cudaEventDestroy(startTransfer);
    cudaEventDestroy(endTransfer);
    cudaEventDestroy(startCompute);
    cudaEventDestroy(endCompute);

    return {computeTime, transferTime};
}

std::pair<double, double> multiplyGPU_async(const std::vector<std::vector<double>>& a,
    const std::vector<std::vector<double>>& b,
    BufferManager& bufferManager) 
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

    auto& buf = bufferManager.getBuffers(aRows, aCols, bCols);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t startTransfer, endTransfer, startCompute, endCompute;
    cudaEventCreate(&startTransfer);
    cudaEventCreate(&endTransfer);
    cudaEventCreate(&startCompute);
    cudaEventCreate(&endCompute);

    cudaEventRecord(startTransfer, stream);
    cudaMemcpyAsync(buf.d_A, A.data(), buf.aSize * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(buf.d_B, B.data(), buf.bSize * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaEventRecord(endTransfer, stream);

    dim3 blockDim(16, 16);
    dim3 gridDim((bCols + 15) / 16, (aRows + 15) / 16);
    cudaEventRecord(startCompute, stream);
    matrixMulKernel<<<gridDim, blockDim, 0, stream>>>(buf.d_A, buf.d_B, buf.d_C, aRows, aCols, bCols);
    cudaEventRecord(endCompute, stream);

    cudaMemcpyAsync(C.data(), buf.d_C, buf.cSize * sizeof(double), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    float transferTime = 0.0f, computeTime = 0.0f;
    cudaEventElapsedTime(&transferTime, startTransfer, endTransfer);
    cudaEventElapsedTime(&computeTime, startCompute, endCompute);

    cudaEventDestroy(startTransfer);
    cudaEventDestroy(endTransfer);
    cudaEventDestroy(startCompute);
    cudaEventDestroy(endCompute);
    cudaStreamDestroy(stream);

    return {computeTime, transferTime};
}

