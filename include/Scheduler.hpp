#pragma once
#include "Balancer.hpp"
#include <thread>
#include <atomic>

class Scheduler 
{
public:
    Scheduler(Balancer& balancer);
    void start();
    void stop();
    void addTask(const Task& task);

private:
    Balancer& balancer;
    std::atomic<bool> running{false};
    std::vector<std::thread> workers;

    void workerThread();
};