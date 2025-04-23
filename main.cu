#include "Analyzer.hpp"
#include "Scheduler.hpp"
#include "Logger.hpp"
#include "WorkerManager.hpp"
#include <memory>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <random>

int GetRandInt(int minNum, int maxNum)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<int> dist(minNum, maxNum);

    return dist(gen);
}

float GetRandFloat(float minNum, float maxNum)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dist(minNum, maxNum);

    return dist(gen);
}

std::vector<float> GetRandMatrix(size_t size) 
{
    std::vector<float> matrix(size * size);
    
    for (auto& val : matrix) {
        val = GetRandFloat(-10, 10);
    }
    
    return matrix;
}

int main() 
{
    auto logger = std::make_shared<Logger>("../log/metrics.json");
    Balancer balancer;
    Scheduler scheduler(balancer, logger);
    
    scheduler.start(4);

    for (int i = 0; i < 3; ++i) 
    {
        size_t size = GetRandInt(500, 524);
        auto matrix = GetRandMatrix(size);

        Analyzer analyzer;
        TaskProfile taskProfile = analyzer.analyze(matrix, size);
        
        Task task
        {
            std::to_string(i),
            taskProfile,
            matrix
        };

        scheduler.addTask(task);
    }

    std::cout << "All tasks submitted. Waiting for completion..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(10));
    scheduler.stop();

    return 0;
}