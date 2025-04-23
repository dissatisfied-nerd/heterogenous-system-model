#pragma once
#include <string>
#include <vector>

enum class DeviceType { CPU, GPU, HYBRID };

struct TaskProfile 
{
    size_t complexity;
    size_t memory_usage;
    DeviceType preferred_device;
    
    bool is_matrix_op = false;
    size_t matrix_size = 0;
};

struct Task 
{
    std::string id;
    TaskProfile profile;
    std::vector<float> data;
};
