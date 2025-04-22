#pragma once
#include "Analyzer.hpp"

struct Task 
{
    std::string id;
    TaskProfile profile;
    std::vector<float> data;
};