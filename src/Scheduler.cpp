#include "Scheduler.hpp"

void Scheduler::addTask(const Task& task) 
{
    std::lock_guard<std::mutex> lock(mtx);
    tasks.push(task);
}

bool Scheduler::hasTasks() 
{
    std::lock_guard<std::mutex> lock(mtx);
    return !tasks.empty();
}

Task Scheduler::getNext() 
{
    std::lock_guard<std::mutex> lock(mtx);
    Task task = tasks.front();
    tasks.pop();
    return task;
}
