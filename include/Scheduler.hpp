#pragma once
#include "Balancer.hpp"
#include "WorkerManager.hpp"
#include <thread>
#include <atomic>
#include <vector>
#include <memory>

class Scheduler 
{
public:
    Scheduler(Balancer& balancer, std::shared_ptr<Logger> logger);
    ~Scheduler();

    void start(size_t numWorkers = 1);
    void stop();
    void addTask(const Task& task);

private:
    Balancer& balancer;
    std::shared_ptr<Logger> logger;
    std::atomic<bool> running{false};
    std::vector<std::thread> workers;

    void workerThread();
};