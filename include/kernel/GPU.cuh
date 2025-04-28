#pragma once
#include "Task.hpp"
#include "memory/GPUPool.cuh"

std::pair<double, double> multiplyGPU(const std::vector<std::vector<double>>& a,
                                      const std::vector<std::vector<double>>& b,
                                      GPUMemoryPool& memoryPool);                                        
