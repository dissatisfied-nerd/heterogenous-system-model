#include "Profiler.hpp"

void Profiler::addSample(const std::string& key, bool isGPU, double timeMs) 
{
    std::lock_guard<std::mutex> lock(mtx);
    auto& stats = isGPU ? gpuStats[key] : cpuStats[key];
    stats.first += 1;
    stats.second += timeMs;
}

double Profiler::averageTime(const std::string& key, bool isGPU) const
{
    std::lock_guard<std::mutex> lock(mtx);
    const auto& stats = isGPU ? gpuStats : cpuStats;
    auto it = stats.find(key);
    if (it == stats.end() || it->second.first == 0) return 1e9;
    return it->second.second / it->second.first;
}

void Profiler::addTransferSample(const std::string& key, double timeMs) 
{
    std::lock_guard<std::mutex> lock(mtx);
    auto& stats = transferStats[key];
    stats.first += 1;
    stats.second += timeMs;
}

double Profiler::averageTransferTime(const std::string& key) const 
{
    std::lock_guard<std::mutex> lock(mtx);
    auto it = transferStats.find(key);
    if (it == transferStats.end() || it->second.first == 0) return 0.0;
    return it->second.second / it->second.first;
}

