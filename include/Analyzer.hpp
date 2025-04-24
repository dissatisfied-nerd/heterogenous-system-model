#pragma once
#include "Task.hpp"
#include "Profiler.hpp"

class Analyzer {
private:
    Profiler& profiler;
    int dynamicThreshold = 64;
    int analysisCounter = 0;
    const int thresholdUpdateInterval = 10;

public:
    Analyzer(Profiler& profiler);
    Task analyze(const std::vector<std::vector<double>>& a,
                    const std::vector<std::vector<double>>& b,
                    int activeCPU, int activeGPU);
    void updateThreshold();
    void loadThreshold(const std::string& filename = "../log/threshold.json");
    void saveThreshold(const std::string& filename = "../log/threshold.json") const;
    int getThreshold() const { return dynamicThreshold; }
};
    
    
