#pragma once
#include "Task.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <chrono>

class Logger 
{
public:
    Logger(const std::string& filename);
    void log(const Task& task, float execTime);
    void logError(const std::string& taskId, const std::string& error);


private:
    std::ofstream logFile;
    nlohmann::json jsonLog;
};
