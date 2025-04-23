cmake_minimum_required(VERSION 3.18)
project(HeterogenousSystemModel LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(nlohmann_json REQUIRED)

file(GLOB SOURCES
    "src/*.cpp"
    "src/kernel/*.cpp"
    "src/kernel/*.cu"
)

add_executable(${PROJECT_NAME} ${SOURCES}
    main.cu)

target_include_directories(${PROJECT_NAME} PRIVATE include)
target_include_directories(${PROJECT_NAME} PRIVATE include/kernel)
target_link_libraries(${PROJECT_NAME} PRIVATE nlohmann_json::nlohmann_json)
