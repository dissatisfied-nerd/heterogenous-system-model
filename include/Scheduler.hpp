#pragma once
#include <queue>
#include <mutex>
#include "Task.hpp"

class Scheduler {
private:
    std::queue<Task> tasks;
    std::mutex mtx;

public:
    void addTask(const Task& task);
    bool hasTasks();
    Task getNext();
};
