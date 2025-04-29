#include "WorkerManager.hpp"
#include "kernel/CPU.hpp"
#include "kernel/GPU.cuh"
#include <fstream>
#include <iostream>

WorkerManager::WorkerManager(Scheduler& sched, Logger& logRef, Profiler& profRef, Analyzer& analyzerRef)
    : scheduler(sched), logger(logRef), profiler(profRef), analyzer(analyzerRef), memoryPool(), bufferManager(memoryPool) {}

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

                size_t bytes = (task.a.size() * task.a[0].size() + task.b.size() * task.b[0].size()) * sizeof(double);
                size_t kb = bytes / 1024;
                std::string key = std::to_string(kb) + "_KB";

                double transferTime = 0.0;
                if (task.useGPU) 
                {
                    activeGPU++;
                    auto [execTime, transferTime] = multiplyGPU_async(task.a, task.b, bufferManager);
                    task.executionTime = execTime;
                    task.transferTime = transferTime;
                    profiler.addSample(key, true, task.executionTime);
                    profiler.addTransferSample(key, task.transferTime);
                    activeGPU--;
                }                
                else 
                {
                    activeCPU++;
                    task.executionTime = multiplyCPU(task.a, task.b);
                    profiler.addSample(key, false, task.executionTime);
                    activeCPU--;
                }
                
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




