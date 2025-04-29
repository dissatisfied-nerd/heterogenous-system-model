#include "Analyzer.hpp"
#include "utils/UUID.hpp"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

Analyzer::Analyzer(Profiler& prof) : profiler(prof) {}

void Analyzer::updateThreshold() 
{
    for (int size = 10; size <= 512; size += 4) 
    {
        std::string key = std::to_string(size) + "x" + std::to_string(size);
        double cpu = profiler.averageTime(key, false);
        double gpu = profiler.averageTime(key, true);
        if (cpu == 1e9 || gpu == 1e9) continue;
        if (gpu < cpu) {
            dynamicThreshold = size;
            break;
        }
    }
}

void Analyzer::saveThreshold(const std::string& filename) const 
{
    std::ofstream out(filename);
    nlohmann::json j;
    j["threshold"] = dynamicThreshold;
    out << j.dump(4);
}

void Analyzer::loadThreshold(const std::string& filename) 
{
    std::ifstream in(filename);
    if (!in || in.peek() == std::ifstream::traits_type::eof()) return;
    nlohmann::json j;
    try {
        in >> j;
        if (j.contains("threshold")) dynamicThreshold = j["threshold"];
    } catch (const nlohmann::json::parse_error& e) {
        std::cerr << "[JSON error] threshold.json: " << e.what() << std::endl;
    }
}


AnalysisResult Analyzer::analyze(const std::vector<std::vector<double>>& a,
                       const std::vector<std::vector<double>>& b,
                       int activeCPU, int activeGPU) 
{
    size_t bytes = (a.size() * a[0].size() + b.size() * b[0].size()) * sizeof(double);
    size_t kb = bytes / 1024;
    std::string key = std::to_string(kb) + "_KB";

    double avgCpu      = profiler.averageTime(key, false);
    double avgGpu      = profiler.averageTime(key, true);
    double avgTransfer = profiler.averageTransferTime(key);

    int matrixSize = std::max(a.size(), b[0].size());
    bool useGPU;

    if (avgCpu == 1e9 && avgGpu == 1e9) {
        useGPU = (matrixSize >= dynamicThreshold);
    }
    else if (avgGpu == 1e9) 
    {
        avgGpu = avgCpu * 0.6;
        avgTransfer = 5.0;
        double cpuLoad = avgCpu * (1 + activeCPU);
        double gpuLoad = (avgGpu + avgTransfer) * (1 + activeGPU);
        useGPU = (gpuLoad < cpuLoad);
    }
    else if (avgCpu == 1e9) {
        useGPU = true;
    }
    else 
    {
        double cpuLoad = avgCpu * (1 + activeCPU);
        double gpuLoad = (avgGpu + avgTransfer) * (1 + activeGPU);
        useGPU = gpuLoad < cpuLoad;
    }

    if (++analysisCounter % thresholdUpdateInterval == 0) {
        updateThreshold();
    }

    std::string reason;
    if (avgCpu == 1e9 && avgGpu == 1e9) {
        reason = "Cold start fallback to threshold";
    } else if (avgGpu == 1e9) {
        reason = "GPU estimate via CPU * 0.6";
    } else if (avgCpu == 1e9) {
        reason = "CPU missing; defaulting to GPU";
    } else {
        reason = useGPU ? "GPU faster" : "CPU faster";
    }

    double cpuLoad = (avgCpu == 1e9) ? 1e9 : avgCpu * (1 + activeCPU);
    double gpuLoad = (avgGpu == 1e9) ? 1e9 : (avgGpu + avgTransfer) * (1 + activeGPU);

    std::string id = UUIDGenerator::generate();
    return AnalysisResult{Task{useGPU, id, a, b}, key, cpuLoad, gpuLoad, avgTransfer, reason};
}



