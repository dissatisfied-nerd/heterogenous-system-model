#include "utils/gen.hpp"

int GetRandInt(int minNum, int maxNum)
{
    static std::random_device rdInt;
    static std::mt19937 genInt(rdInt());
    static std::uniform_int_distribution<int> distInt(minNum, maxNum);

    return distInt(genInt);
}

double GetRandDouble(double minNum, double maxNum)
{
    static std::random_device rdDouble;
    static std::mt19937 genDouble(rdDouble());
    static std::uniform_real_distribution<double> distDouble(minNum, maxNum);

    return distDouble(genDouble);
}

std::vector<std::vector<double>> GetRandMatrix(int n, int k)
{
    std::vector<std::vector<double>> res(n, std::vector<double>(k));

    for (int i = 0; i < n; ++i){
        for (int j = 0; j < k; ++j){
            res[i][j] = GetRandDouble(-10, 10);
        }
    }

    return res;
}