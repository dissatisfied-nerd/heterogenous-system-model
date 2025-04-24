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

void Analyzer::loadThreshold(const std::string& filename) {
    std::ifstream in(filename);
    if (!in || in.peek() == std::ifstream::traits_type::eof()) return; // файл не существует или пуст
    nlohmann::json j;
    try {
        in >> j;
        if (j.contains("threshold")) dynamicThreshold = j["threshold"];
    } catch (const nlohmann::json::parse_error& e) {
        std::cerr << "[JSON error] threshold.json: " << e.what() << std::endl;
    }
}


Task Analyzer::analyze(const std::vector<std::vector<double>>& a,
                       const std::vector<std::vector<double>>& b,
                       int activeCPU, int activeGPU) {
    std::string key = std::to_string(a.size()) + "x" + std::to_string(b[0].size());

    double avgCpu = profiler.averageTime(key, false);
    double avgGpu = profiler.averageTime(key, true);
    int matrixSize = std::max(a.size(), b[0].size());

    bool useGPU;

    if (avgCpu == 1e9 && avgGpu == 1e9) {
        useGPU = matrixSize >= dynamicThreshold;
    } else {
        double predictedCpu = avgCpu * (1 + activeCPU);
        double predictedGpu = avgGpu * (1 + activeGPU);
        useGPU = predictedGpu < predictedCpu;
    }

    // каждые N задач — обновляем порог
    if (++analysisCounter % thresholdUpdateInterval == 0) {
        updateThreshold();
    }

    return Task{useGPU, UUIDGenerator::generate(), a, b};
}


