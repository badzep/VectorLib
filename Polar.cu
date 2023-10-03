#include <cmath>


#include "Polar.cuh"
#include "Math.cuh"


template<typename T>
__host__ __device__ Polar<T>::Polar() {}
template<typename T>
__host__ __device__ Polar<T>::Polar(const T magnitude, const T angle): magnitude(magnitude), angle(angle) {}
template<typename T>
__host__ __device__ Polar<T> Polar<T>::wrap_angle() const {
    T new_angle = fmod(angle + (T) PI, (T) TAU);
    return {this->magnitude, this->angle + (T) (new_angle < (T) 0.0 ? (T) PI : -(T) PI)};
}
template<typename T>
__host__ __device__ void Polar<T>::wrap_angle_in_place() {
    this->angle = fmod(this->angle + (T) PI, (T) TAU);
    this->angle += (T) (this->angle < (T) 0.0 ? (T) PI : -(T) PI);
}
