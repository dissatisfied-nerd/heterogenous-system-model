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

    size_t bytes = (task.a.size() * task.a[0].size() +
                    task.b.size() * task.b[0].size()) * sizeof(double);
    size_t kb = bytes / 1024;

    data[task.id] = {
        {"architecture", task.useGPU ? "GPU" : "CPU"},
        {"executionTime_ms", task.executionTime},
        {"data_volume_kb", kb}
    };
    if (task.useGPU) {
        data[task.id]["transfer_time_ms"] = task.transferTime;
    }
}


void Logger::writeToFile(const std::string& filename) const 
{
    std::string actualFilename = + filename.empty() ? "../log/regular/" + getTimestamp() + ".json" : filename;
    std::ofstream out(actualFilename);
    out << data.dump(4);
}

void Logger::logSummary(const Profiler& profiler, int threshold) 
{
    std::lock_guard<std::mutex> lock(mtx);
    nlohmann::json summary;

    for (auto& [id, entry] : data.items()) {
        if (!entry.contains("data_volume_kb") ||
            !entry.contains("executionTime_ms")) continue;

        auto kb  = entry["data_volume_kb"].get<size_t>();
        std::string key = std::to_string(kb) + "_KB";

        double cpuAvg     = profiler.averageTime(key, false);
        double gpuAvg     = profiler.averageTime(key, true);
        double transferAvg = profiler.averageTransferTime(key);

        summary[key] = {
            {"avg_CPU_ms", cpuAvg},
            {"avg_GPU_ms", gpuAvg},
            {"avg_transfer_ms", transferAvg}
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

void Logger::logDecision(const std::string& taskId,
    const std::string& matrixKey,
    double cpuLoad,
    double gpuLoad,
    double avgTransfer,
    const std::string& reason) 
{
    std::lock_guard<std::mutex> lock(mtx);

    if (!data.contains(taskId)) {
        data[taskId] = nlohmann::json::object();
    }

    data[taskId]["decision_debug"] = {
        {"matrix_size", matrixKey},
        {"predicted_CPU_ms", cpuLoad},
        {"predicted_GPU_ms", gpuLoad},
        {"avg_transfer_ms", avgTransfer},
        {"decision_reason", reason}
    };
}


