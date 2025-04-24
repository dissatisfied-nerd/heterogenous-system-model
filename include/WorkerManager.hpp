#pragma once
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <nlohmann/json.hpp>
#include "Scheduler.hpp"
#include "Logger.hpp"

class WorkerManager 
{
private:
    std::vector<std::thread> workers;
    Scheduler& scheduler;

    Logger& logger;
    std::atomic<int> activeTasks = 0;
    std::mutex syncMutex;
    std::condition_variable syncCv;

public:
    WorkerManager(Scheduler& sched, Logger& logRef);
    void start(int numThreads);
    void wait();
};

