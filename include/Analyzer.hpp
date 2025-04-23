#pragma once
#include <string>
#include <vector>
#include "Task.hpp"

class Analyzer 
{
public:
    TaskProfile analyze(const std::vector<float>& input_data, size_t data_size);
};