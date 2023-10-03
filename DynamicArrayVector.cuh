#ifndef VECTORLIB_DYNAMICARRAYVECTOR_CUH
#define VECTORLIB_DYNAMICARRAYVECTOR_CUH

/**
 * An implementation of std::vector for cuda
 * @tparam T
 * @tparam INITIAL_CAPACITY
 */
template<typename T, const unsigned int INITIAL_CAPACITY>
class vector {
protected:
    unsigned int length = 0;
    unsigned int capacity = INITIAL_CAPACITY;
    T* array;

    __device__ void reallocate();

public:
    __host__ __device__ vector();

    __host__ __device__ ~vector();

    __device__ T& operator[](unsigned int index);

    __device__ void push_back(T value);

    __device__ unsigned int get_length() const;

    __device__ unsigned int get_capacity() const;
    // todo copy to and copy from
};


#endif //VECTORLIB_DYNAMICARRAYVECTOR_CUH
