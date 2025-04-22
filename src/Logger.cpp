// src/Logger.cpp
#include "Logger.hpp"

Logger::Logger(const std::string& filename) {
    logFile.open(filename);
}

void Logger::log(const Task& task, float execTime) 
{
    nlohmann::json entry;
    entry["task_id"] = task.id;
    entry["device"] = (task.profile.preferred_device == DeviceType::CPU) ? "CPU" : "GPU";
    entry["exec_time_ms"] = execTime;
    entry["timestamp"] = std::chrono::system_clock::now().time_since_epoch().count();
    
    jsonLog.push_back(entry);
    logFile << jsonLog.dump(4) << std::endl;
}