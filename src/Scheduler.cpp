#include "Scheduler.hpp"
#include <iostream>
#include <chrono>

Scheduler::Scheduler(Balancer& balancer, std::shared_ptr<Logger> logger) 
    : balancer(balancer), logger(std::move(logger)) {}

Scheduler::~Scheduler() {
    stop();
}

void Scheduler::start(size_t numWorkers) 
{
    if (running){
        return;
    }
    
    running = true;
    
    for (size_t i = 0; i < numWorkers; ++i) {
        workers.emplace_back(&Scheduler::workerThread, this);
    }
}

void Scheduler::stop() 
{
    if (!running) {
        return;
    }
    
    running = false;
    
    for (auto& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    
    workers.clear();
}

void Scheduler::addTask(const Task& task) {
    balancer.addTask(task);
}

void Scheduler::workerThread() 
{
    WorkerManager workerManager(logger);
    
    while (running) 
    {
        Task task = balancer.getNextTask();
        
        if (task.id.empty()) 
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        try {
            workerManager.executeTask(task);  // Теперь вызов корректен
        } 
        catch (const std::exception& e) {
            std::cerr << "Error executing task " << task.id << ": " << e.what() << std::endl;
        }
    }
}