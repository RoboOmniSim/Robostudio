cmake_minimum_required(VERSION 3.18)
project(barycentric LANGUAGES CXX CUDA)

# Set the CUDA standard
set(CMAKE_CUDA_STANDARD 14)

# Find pybind11
find_package(pybind11 REQUIRED)

# Add CUDA and C++ files
add_library(barycentric SHARED
    barycentric.cu
    barycentric_binding.cpp
)

# Link pybind11
target_link_libraries(barycentric PRIVATE pybind11::module)

# Specify include directories
target_include_directories(barycentric PRIVATE ${PYBIND11_INCLUDE_DIRS})