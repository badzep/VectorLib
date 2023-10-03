#include "DynamicArrayVector.cuh"


template<typename T, const unsigned int INITIAL_CAPACITY>
__device__ void vector<T, INITIAL_CAPACITY>::reallocate() {
    this->capacity *= 2;
    T* new_array;
    cudaMalloc(&new_array, this->capacity * sizeof(T));
    cudaFree(this->array);
    this->array = new_array;
}

template<typename T, const unsigned int INITIAL_CAPACITY>
__host__ __device__ vector<T, INITIAL_CAPACITY>::vector() {
    cudaMalloc(&this->array, this->capacity);
}

template<typename T, const unsigned int INITIAL_CAPACITY>
__host__ __device__ vector<T, INITIAL_CAPACITY>::~vector() {
    cudaFree(this->array);
}

template<typename T, const unsigned int INITIAL_CAPACITY>
__device__ T &vector<T, INITIAL_CAPACITY>::operator[](unsigned int index) {
    return this->array[index];
}

template<typename T, const unsigned int INITIAL_CAPACITY>
__device__ void vector<T, INITIAL_CAPACITY>::push_back(T value) {
    if (this->length == capacity) {
        this->reallocate();
    }
    this->array[this->length] = value;
    this->length++;
}

template<typename T, const unsigned int INITIAL_CAPACITY>
__device__ unsigned int vector<T, INITIAL_CAPACITY>::get_length() const {
    return this->length;
}

template<typename T, const unsigned int INITIAL_CAPACITY>
__device__ unsigned int vector<T, INITIAL_CAPACITY>::get_capacity() const {
    return this->capacity;
}