#include "Balancer.hpp"

void Balancer::addTask(const Task& task) 
{
    std::lock_guard<std::mutex> lock(mtx);
    
    if (task.profile.preferred_device == DeviceType::CPU) {
        cpuQueue.push(task);
    } 
    else {
        gpuQueue.push(task);
    }
}

Task Balancer::getNextTask() 
{
    std::lock_guard<std::mutex> lock(mtx);
    
    if (cpuLoad < gpuLoad && !cpuQueue.empty()) {
        cpuLoad += 0.1f;
        auto task = cpuQueue.front();
        cpuQueue.pop();
        return task;
    } 
    else if (!gpuQueue.empty()) {
        gpuLoad += 0.1f;
        auto task = gpuQueue.front();
        gpuQueue.pop();
        return task;
    }
    
    return Task{};  // Пустая задача, если очередь пуста
}