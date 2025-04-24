#include "Analyzer.hpp"
#include "Scheduler.hpp"
#include "WorkerManager.hpp"
#include "utils/gen.hpp"

int main() 
{
    Analyzer analyzer;
    Scheduler scheduler;
    Logger logger;

    for (int i = 0; i < 10; ++i) 
    {
        int size = GetRandInt(2, 100);
        std::vector<std::vector<double>> a = GetRandMatrix(size, size);
        std::vector<std::vector<double>> b = GetRandMatrix(size, size);

        scheduler.addTask(analyzer.analyze(a, b));
    }

    WorkerManager manager(scheduler, logger);
    manager.start(4);
    manager.wait();

    return 0;
}
