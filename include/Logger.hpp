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

private:
    std::ofstream logFile;
    nlohmann::json jsonLog;
};