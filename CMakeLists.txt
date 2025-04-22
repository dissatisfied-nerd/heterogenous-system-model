cmake_minimum_required(VERSION 3.12)
project(HeterogeneousComputing LANGUAGES CXX CUDA)  # Поддержка CUDA

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "native")  # Автовыбор архитектуры GPU

# Поиск необходимых библиотек
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)

# Подключение JSON-библиотеки (например, nlohmann/json)
include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(json)

# Основной проект
add_executable(hcomp
    src/main.cpp
    src/Analyzer.cpp
    src/Balancer.cpp
    src/Scheduler.cpp
    src/Logger.cpp
)

target_include_directories(hcomp PRIVATE include)
target_link_libraries(hcomp PRIVATE OpenMP::OpenMP_CXX CUDA::cudart nlohmann_json::nlohmann_json)

# Тесты (опционально)
enable_testing()
add_executable(test_analyzer tests/test_analyzer.cpp src/Analyzer.cpp)
target_link_libraries(test_analyzer PRIVATE hcomp)
add_test(NAME test_analyzer COMMAND test_analyzer)