#include "Logger.hpp"
#include "Profiler.hpp"
#include <fstream>
#include <iostream>

std::string getTimestamp() 
{
    auto now = std::chrono::system_clock::now();
    std::time_t t_c = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&t_c);

    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d_%H-%M-%S", &tm);
    return std::string(buffer);
}

void Logger::logTask(const Task& task) {
    std::lock_guard<std::mutex> lock(mtx);

    std::string key = std::to_string(task.a.size()) + "x" + std::to_string(task.b[0].size());

    data[task.id] = {
        {"architecture", task.useGPU ? "GPU" : "CPU"},
        {"executionTime_ms", task.executionTime},
        {"matrix_size", key}
    };
}

void Logger::writeToFile(const std::string& filename) const 
{
    std::string actualFilename = + filename.empty() ? "../log/regular/" + getTimestamp() + ".json" : filename;
    std::ofstream out(actualFilename);
    out << data.dump(4);
}

void Logger::logSummary(const Profiler& profiler, int threshold) {
    std::lock_guard<std::mutex> lock(mtx);
    nlohmann::json summary;

    for (const auto& [taskId, entry] : data.items()) {
        if (!entry.contains("architecture") || !entry.contains("executionTime_ms") || !entry.contains("matrix_size"))
            continue;

        std::string key = entry["matrix_size"];
        double cpuAvg = profiler.averageTime(key, false);
        double gpuAvg = profiler.averageTime(key, true);

        summary[key] = {
            {"avg_CPU_ms", cpuAvg},
            {"avg_GPU_ms", gpuAvg}
        };
    }

    std::string timestamp = getTimestamp();

    nlohmann::json fullSummary;
    std::ifstream in("../log/summary.json");
    if (in && in.peek() != std::ifstream::traits_type::eof()) {
        try {
            in >> fullSummary;
        } catch (const nlohmann::json::parse_error& e) {
            std::cerr << "[JSON error] summary.json: " << e.what() << std::endl;
        }
    }

    fullSummary[timestamp] = summary;
    fullSummary[timestamp]["threshold"] = threshold;

    std::ofstream out("../log/summary.json");
    out << fullSummary.dump(4);
}



