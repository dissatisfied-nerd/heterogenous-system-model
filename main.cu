#include "Scheduler.hpp"
#include "Logger.hpp"
#include <vector>

int main() 
{
    Balancer balancer;
    Scheduler scheduler(balancer);
    Logger logger("metrics.json");

    scheduler.start();

    std::vector<float> bigData(10000, 1.0f);
    std::vector<float> smallData(100, 1.0f);

    Task task1{"task1", Analyzer().analyze(bigData), bigData};
    Task task2{"task2", Analyzer().analyze(smallData), smallData};

    scheduler.addTask(task1);
    scheduler.addTask(task2);

    std::this_thread::sleep_for(std::chrono::seconds(2));
    scheduler.stop();

    return 0;
}