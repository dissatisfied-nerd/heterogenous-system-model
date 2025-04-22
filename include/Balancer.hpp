// include/Balancer.hpp
#pragma once
#include "Task.hpp"
#include <queue>
#include <mutex>

class Balancer 
{
public:
    void addTask(const Task& task);
    Task getNextTask();
    float getCpuLoad() const;
    float getGpuLoad() const;

private:
    std::queue<Task> cpuQueue;
    std::queue<Task> gpuQueue;
    mutable std::mutex mtx;
    float cpuLoad = 0.0f;
    float gpuLoad = 0.0f;
};