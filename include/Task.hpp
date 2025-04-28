#pragma once
#include <vector>
#include <string>

struct Task {
    bool useGPU;
    std::string id;
    std::vector<std::vector<double>> a;
    std::vector<std::vector<double>> b;
    double executionTime = 0.0;
    double transferTime = 0.0;
};
