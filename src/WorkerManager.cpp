#include "WorkerManager.hpp"
#include "kernel/CPU.hpp"
#include "kernel/GPU.cuh"
#include <fstream>
#include <iostream>

WorkerManager::WorkerManager(Scheduler& sched, Logger& logRef, Profiler& profRef, Analyzer& analyzerRef)
    : scheduler(sched), logger(logRef), profiler(profRef), analyzer(analyzerRef) {}

void WorkerManager::start(int numThreads) 
{
    for (int i = 0; i < numThreads; ++i) 
    {
        workers.emplace_back([this]() 
        {
            while (true) 
            {
                Task task;
                {
                    std::unique_lock<std::mutex> lock(syncMutex);
                    if (!scheduler.hasTasks()) {
                        break;
                    }
        
                    task = scheduler.getNext();
                    ++activeTasks;
                }

                if (task.useGPU) 
                {
                    activeGPU++;
                    task.executionTime = multiplyGPU(task);
                    activeGPU--;
                } 
                else 
                {
                    activeCPU++;
                    task.executionTime = multiplyCPU(task);
                    activeCPU--;
                }
                

                std::string key = std::to_string(task.a.size()) + "x" + std::to_string(task.b[0].size());
                profiler.addSample(key, task.useGPU, task.executionTime);
                logger.logTask(task);

                {
                    std::lock_guard<std::mutex> lock(syncMutex);
                    --activeTasks;
                    
                    if (activeTasks == 0 && !scheduler.hasTasks()) {
                        syncCv.notify_one();
                    }
                }
            }
        });
    }
}

void WorkerManager::wait() {
    {
        std::unique_lock<std::mutex> lock(syncMutex);
        syncCv.wait(lock, [this]() {
            return activeTasks == 0 && !scheduler.hasTasks();
        });
    }

    for (auto& worker : workers)
        if (worker.joinable()) worker.join();

    logger.logSummary(profiler, analyzer.getThreshold());
    analyzer.saveThreshold();

    logger.writeToFile();
}




