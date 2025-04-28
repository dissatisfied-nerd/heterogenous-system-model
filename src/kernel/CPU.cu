#include "kernel/CPU.hpp"
#include <chrono>

double multiplyCPU(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b)
{
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<double>> result(a.size(), std::vector<double>(b[0].size(), 0));
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b[0].size(); ++j)
            for (size_t k = 0; k < a[0].size(); ++k)
                result[i][j] += a[i][k] * b[k][j];

    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count();
}
