cmake_minimum_required(VERSION 3.26)
project(VectorLib CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_library(VectorLib
        VectorLib.cuh
        Math.cuh
        DynamicArrayVector.cuh
        DynamicArrayVector.cu
        Vector.cuh
        Vector.cu
        Polar.cuh
        Polar.cu
)

set_target_properties(VectorLib PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
