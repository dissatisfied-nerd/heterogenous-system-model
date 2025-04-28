#pragma once
#include <mutex>
#include <unordered_map>
#include <string>

class Profiler {
private:
    mutable std::mutex mtx;
    std::unordered_map<std::string, std::pair<int, double>> cpuStats;
    std::unordered_map<std::string, std::pair<int, double>> gpuStats;
    std::unordered_map<std::string, std::pair<int, double>> transferStats;

public:
    void addSample(const std::string& key, bool isGPU, double timeMs);
    double averageTime(const std::string& key, bool isGPU) const;

    void addTransferSample(const std::string& key, double timeMs);
    double averageTransferTime(const std::string& key) const;
};
