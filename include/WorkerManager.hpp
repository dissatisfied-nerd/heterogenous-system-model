#pragma once
#include "Task.hpp"
#include "kernel/CPU.hpp"
#include "kernel/GPU.cuh"
#include "Logger.hpp"
#include <memory>

class WorkerManager 
{
public:
    explicit WorkerManager(std::shared_ptr<Logger> logger);
    
    void executeTask(const Task& task);

    WorkerManager(const WorkerManager&) = delete;
    WorkerManager& operator=(const WorkerManager&) = delete;

private:
    std::shared_ptr<Logger> logger;
    
    void executeOnCpu(const Task& task);
    void executeOnGpu(const Task& task);
};