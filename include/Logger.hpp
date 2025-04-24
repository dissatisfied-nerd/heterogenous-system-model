#pragma once
#include <mutex>
#include <nlohmann/json.hpp>
#include "Task.hpp"
#include "Profiler.hpp"

std::string getTimestamp();

class Logger {
private:
    std::mutex mtx;
    nlohmann::json data;

public:
    void logTask(const Task& task);
    void writeToFile(const std::string& filename = "") const;
    void logSummary(const Profiler& profiler, int threshold);
};
