// src/Scheduler.cpp
#include "Scheduler.hpp"
#include <iostream>

Scheduler::Scheduler(Balancer& balancer) : balancer(balancer) {}

void Scheduler::start() 
{
    running = true;
    workers.emplace_back(&Scheduler::workerThread, this);
}

void Scheduler::stop() 
{
    running = false;
    for (auto& worker : workers) {
        if (worker.joinable()) worker.join();
    }
}

void Scheduler::workerThread() 
{
    while (running) 
    {
        Task task = balancer.getNextTask();

        if (task.id.empty()) 
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        std::cout << "Executing task: " << task.id << std::endl;
    }
}