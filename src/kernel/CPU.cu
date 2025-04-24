#include "kernel/CPU.hpp"
#include <chrono>

double multiplyCPU(const Task& task) 
{
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<double>> result(task.a.size(), std::vector<double>(task.b[0].size(), 0));
    for (size_t i = 0; i < task.a.size(); ++i)
        for (size_t j = 0; j < task.b[0].size(); ++j)
            for (size_t k = 0; k < task.a[0].size(); ++k)
                result[i][j] += task.a[i][k] * task.b[k][j];

    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count();
}
