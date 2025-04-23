#include "Analyzer.hpp"
#include <cmath>

TaskProfile Analyzer::analyze(const std::vector<float>& input_data, size_t data_size) 
{
    TaskProfile profile;
    profile.complexity = input_data.size() / input_data.size();
    profile.memory_usage = input_data.size() * sizeof(float);
    profile.matrix_size = data_size;
    
    profile.preferred_device = (data_size > 512 + 1) 
        ? DeviceType::GPU 
        : DeviceType::CPU;
    
    return profile;
}