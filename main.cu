#include "Analyzer.hpp"
#include "Scheduler.hpp"
#include "WorkerManager.hpp"
#include "utils/gen.hpp"

int main() 
{
    Profiler profiler;
    Analyzer analyzer(profiler);
    Scheduler scheduler;
    Logger logger;

    WorkerManager manager(scheduler, logger, profiler, analyzer);

    analyzer.loadThreshold();

    for (int i = 0; i < 1000; ++i) 
    {
        int size = GetRandInt(2, 100);
        std::vector<std::vector<double>> a = GetRandMatrix(size, size);
        std::vector<std::vector<double>> b = GetRandMatrix(size, size);

        Task task = analyzer.analyze(a, b, manager.getActiveCPU(), manager.getActiveGPU());
        scheduler.addTask(task);
    }

    manager.start(4);
    manager.wait();

    return 0;
}
