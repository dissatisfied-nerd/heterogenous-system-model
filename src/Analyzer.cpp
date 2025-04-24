#include "Analyzer.hpp"
#include "utils/UUID.hpp"

Task Analyzer::analyze(const std::vector<std::vector<double>>& a,
                       const std::vector<std::vector<double>>& b) 
{
    bool useGPU = (a.size() > 64);
    return Task{useGPU, UUIDGenerator::generate(), a, b};
}
