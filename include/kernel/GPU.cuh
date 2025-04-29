#pragma once
#include "Task.hpp"
#include "memory/BufferManager.cuh"

std::pair<double, double> multiplyGPU(const std::vector<std::vector<double>>& a,
    const std::vector<std::vector<double>>& b,
    BufferManager& bufferManager);
                                    
std::pair<double, double> multiplyGPU_async(const std::vector<std::vector<double>>& a,
    const std::vector<std::vector<double>>& b,
    BufferManager& bufferManager);