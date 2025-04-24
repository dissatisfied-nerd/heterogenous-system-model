#pragma once
#include <mutex>
#include <nlohmann/json.hpp>
#include "Task.hpp"

class Logger {
private:
    std::mutex mtx;
    nlohmann::json data;

public:
    void logTask(const Task& task);
    void writeToFile(const std::string& filename) const;
};
