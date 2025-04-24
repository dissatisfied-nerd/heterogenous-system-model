#pragma once
#include "Task.hpp"

class Analyzer {
public:
    Task analyze(const std::vector<std::vector<double>>& a,
                 const std::vector<std::vector<double>>& b);
};
