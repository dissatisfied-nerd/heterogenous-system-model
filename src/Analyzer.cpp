#include "Analyzer.hpp"
#include <cmath>

TaskProfile Analyzer::analyze(const std::vector<float>& input_data) 
{
    TaskProfile profile;
    profile.complexity = input_data.size();
    profile.memory_usage = input_data.size() * sizeof(float);
    
    // Эвристика: если данных много → GPU, если мало → CPU
    profile.preferred_device = (input_data.size() > 1024) 
        ? DeviceType::GPU 
        : DeviceType::CPU;
    
    return profile;
}