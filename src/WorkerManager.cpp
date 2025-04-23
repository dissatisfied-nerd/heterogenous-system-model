#include "WorkerManager.hpp"
#include <chrono>
#include <stdexcept>
#include <cmath>
#include <iostream>

WorkerManager::WorkerManager(std::shared_ptr<Logger> logger) 
    : logger(std::move(logger)) {}

void WorkerManager::executeTask(const Task& task) 
{
    auto start = std::chrono::high_resolution_clock::now();
    
    try 
    {
        switch(task.profile.preferred_device) 
        {
            case DeviceType::CPU:
                executeOnCpu(task);
                break;
            
            case DeviceType::GPU:
                executeOnGpu(task);
                break;
            
            default:
                throw std::runtime_error("Unsupported device type");
        }
    } 
    catch (const std::exception& e) 
    {
        logger->logError(task.id, e.what());
        throw;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    float duration = std::chrono::duration<float, std::milli>(end - start).count();
    logger->log(task, duration);
}

void WorkerManager::executeOnCpu(const Task& task) 
{
    const size_t data_size = task.data.size();

    size_t matrix_size = static_cast<size_t>(std::sqrt(data_size));

    if (matrix_size * matrix_size != data_size) {
        throw std::runtime_error("Invalid matrix size");
    }
    
    std::vector<float> result(data_size);
    CpuKernel::matrixMul(task.data, task.data, result, matrix_size);
}

void WorkerManager::executeOnGpu(const Task& task) 
{
    const size_t data_size = task.data.size();

    size_t matrix_size = static_cast<size_t>(std::sqrt(data_size));

    if (matrix_size * matrix_size != data_size) {
        throw std::runtime_error("Invalid matrix size");
    }
    
    float *result = (float*)malloc(data_size * sizeof(float));
    CudaKernel::matrixMul(task.data, task.data, result, matrix_size);
}