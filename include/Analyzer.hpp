#pragma once
#include <string>
#include <vector>

enum class DeviceType { CPU, GPU, HYBRID };

struct TaskProfile 
{
    size_t complexity;
    size_t memory_usage;
    DeviceType preferred_device;
};

class Analyzer 
{
public:
    TaskProfile analyze(const std::vector<float>& input_data);
};