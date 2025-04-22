#!/bin/bash

clean_found=false
test_found=false

for arg in "$@"; do
    case "$arg" in
        --clean)
            clean_found=true
            ;;
        --test)
            test_found=true
            ;;
    esac
done

if $clean_found; then
    rm -rf build
fi

mkdir -p build
cd build

cmake ..
cmake --build .


if $test_found; then
    ctest
else
    ./HeterogenousSystem
fi

