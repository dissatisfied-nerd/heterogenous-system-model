#pragma once
#include "Task.hpp"

struct AnalysisResult {
    Task task;
    std::string key;
    double cpuLoad;
    double gpuLoad;
    double transfer;
    std::string reason;
};
