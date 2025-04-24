#include "WorkerManager.hpp"
#include "kernel/CPU.hpp"
#include "kernel/GPU.cuh"
#include <fstream>
#include <iostream>

WorkerManager::WorkerManager(Scheduler& sched, Logger& logRef)
    : scheduler(sched), logger(logRef) {}

void WorkerManager::start(int numThreads) {
    for (int i = 0; i < numThreads; ++i) {
        workers.emplace_back([this]() {
            while (true) {
                Task task;
                {
                    std::unique_lock<std::mutex> lock(syncMutex);
                    if (!scheduler.hasTasks()) break;
                    task = scheduler.getNext();
                    ++activeTasks;
                }

                task.executionTime = task.useGPU ? multiplyGPU(task) : multiplyCPU(task);
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

    logger.writeToFile("../log/metrics.json");
}


