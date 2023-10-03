#ifndef VECTORLIB_POLAR_CUH
#define VECTORLIB_POLAR_CUH

/**
 * Polar coordinates
 */
template<typename T>
class Polar {
public:
    T magnitude;
    T angle;

    __host__ __device__ Polar();
    __host__ __device__ Polar(T magnitude,T angle);
    __host__ __device__ ~Polar() = default;
    __host__ __device__ Polar<T> wrap_angle() const;
    __host__ __device__ void wrap_angle_in_place();
};

/**
 * Polar coordinates as float
 */
using Polarf = Polar<float>;

/**
 * Polar coordinates as double
 */
using Polard = Polar<double>;

template class Polar<float>;
template class Polar<double>;

#endif //VECTORLIB_POLAR_CUH
