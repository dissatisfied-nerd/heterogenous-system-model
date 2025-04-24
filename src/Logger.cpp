#include "Logger.hpp"
#include <fstream>

void Logger::logTask(const Task& task) 
{
    std::lock_guard<std::mutex> lock(mtx);
    data[task.id] = {
        {"architecture", task.useGPU ? "GPU" : "CPU"},
        {"matrixSize", task.a.size()},
        {"executionTime_ms", task.executionTime}
    };
}

void Logger::writeToFile(const std::string& filename) const 
{
    std::ofstream out(filename);
    out << data.dump(4);
}
