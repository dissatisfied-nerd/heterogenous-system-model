#pragma once
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <nlohmann/json.hpp>
#include "Scheduler.hpp"
#include "Logger.hpp"
#include "Profiler.hpp"
#include "Analyzer.hpp"
#include "memory/GPUPool.cuh"
#include "memory/BufferManager.cuh"

class WorkerManager 
{
private:
    std::vector<std::thread> workers;
    
    Scheduler& scheduler;
    Logger& logger;
    Profiler& profiler;
    Analyzer& analyzer;

    GPUMemoryPool memoryPool;
    BufferManager bufferManager;

    std::atomic<int> activeTasks = 0;
    std::mutex syncMutex;
    std::condition_variable syncCv;

    std::atomic<int> activeCPU = 0;
    std::atomic<int> activeGPU = 0;

public:
    WorkerManager(Scheduler& sched, Logger& logRef, Profiler& profRef, Analyzer& analyzerRef);
    void start(int numThreads);
    void wait();

    int getActiveCPU() const { return activeCPU.load(); }
    int getActiveGPU() const { return activeGPU.load(); }
};

