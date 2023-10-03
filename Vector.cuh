#ifndef VECTORLIB_VECTOR_CUH
#define VECTORLIB_VECTOR_CUH

#include "Polar.cuh"

/**
 * Cartesian coordinates
 */
template<const unsigned char DIMENSIONS, typename T>
class Vec;

template<typename T>
using Vec2 = Vec<2, T>;
template<typename T>
using Vec2 = Vec<2, T>;

template<typename T>
using Vec3 = Vec<3, T>;
template<typename T>
using Vec3 = Vec<3, T>;

using Vec2f = Vec<2, float>;
using Vec3f = Vec<3, float>;

using Vec2d = Vec<2, double>;
using Vec3d = Vec<3, double>;

template<typename T>
class Vec<2, T> {
public:
    T x;
    T y;
    
    __host__ __device__ Vec();
    __host__ __device__ Vec(T x, T y);
    __host__ __device__ Vec(const Vec<2, T>& other);
    __host__ __device__ ~Vec();

    __host__ __device__ Vec<2, T> operator+(Vec<2, T> other_vector) const;
    __host__ __device__ Vec<2, T> operator+(T other) const;
    __host__ __device__ void operator+=(Vec<2, T> other_vector);
    __host__ __device__ void operator+=(T other);
    __host__ __device__ Vec<2, T> add(Vec<2, T> other_vector) const;
    __host__ __device__ Vec<2, T> add(T other) const;
    __host__ __device__ void add_in_place(Vec<2, T> other_vector);
    __host__ __device__ void add_in_place(T other);

    __host__ __device__ Vec<2, T> operator-(Vec<2, T> other_vector) const;
    __host__ __device__ Vec<2, T> operator-(T other) const;
    __host__ __device__ void operator-=(Vec<2, T> other_vector);
    __host__ __device__ void operator-=(T other);
    __host__ __device__ Vec<2, T> subtract(Vec<2, T> other_vector) const;
    __host__ __device__ Vec<2, T> subtract(T other) const;
    __host__ __device__ void subtract_in_place(Vec<2, T> other_vector);
    __host__ __device__ void subtract_in_place(T other);

    __host__ __device__ Vec<2, T> operator*(Vec<2, T> other_vector) const;
    __host__ __device__ Vec<2, T> operator*(T other) const;
    __host__ __device__ void operator*=(Vec<2, T> other_vector);
    __host__ __device__ void operator*=(T other);
    __host__ __device__ Vec<2, T> multiply(Vec<2, T> other_vector) const;
    __host__ __device__ Vec<2, T> multiply(T other) const;
    __host__ __device__ void multiply_in_place(Vec<2, T> other_vector);
    __host__ __device__ void multiply_in_place(T other);

    __host__ __device__ Vec<2, T> operator/(Vec<2, T> other_vector) const;
    __host__ __device__ Vec<2, T> operator/(T other) const;
    __host__ __device__ void operator/=(Vec<2, T> other_vector);
    __host__ __device__ void operator/=(T other);
    __host__ __device__ Vec<2, T> divide(Vec<2, T> other_vector) const;
    __host__ __device__ Vec<2, T> divide(T other) const;
    __host__ __device__ void divide_in_place(Vec<2, T> other_vector);
    __host__ __device__ void divide_in_place(T other);

    __host__ __device__ Vec<2, T> power(Vec<2, T> other_vector) const;
    __host__ __device__ Vec<2, T> power(T other) const;
    __host__ __device__ void power_in_place(Vec<2, T> other_vector);
    __host__ __device__ void power_in_place(T other);

    __host__ __device__ Vec<2, T> abs() const;
    __host__ __device__ void abs_in_place();

    __host__ __device__ Vec<2, T> sqrt() const;
    __host__ __device__ void sqrt_in_place();

    __host__ __device__ Vec<2, T> round() const;
    __host__ __device__ void round_in_place();

    __host__ __device__ T get_distance_to(Vec<2, T> other_vector) const;
    __host__ __device__ T magnitude() const;

    __host__ __device__ T angle() const;
    __host__ __device__ Vec<2, T> normalize() const;
    __host__ __device__ void normalize_in_place();

    __host__ __device__ Vec<2, T> rotate(T angle) const;
    __host__ __device__ void rotate_in_place(T angle);

    __host__ __device__ Vec<2, T> min(T minimum) const;
    __host__ __device__ Vec<2, T> min(Vec<2, T> minimum) const;
    __host__ __device__ Vec<2, T> max(T maximum) const;
    __host__ __device__ Vec<2, T> max(Vec<2, T> maximum) const;

    __host__ __device__ void min_in_place(T minimum);
    __host__ __device__ void min_in_place(Vec<2, T> minimum);
    __host__ __device__ void max_in_place(T maximum);
    __host__ __device__ void max_in_place(Vec<2, T> maximum);

    __host__ __device__ Vec<2, T> range(T minimum, T maximum) const;
    __host__ __device__ Vec<2, T> range(Vec<2, T> minimum, Vec<2, T> maximum) const;
    __host__ __device__ void range_in_place(T minimum, T maximum);
    __host__ __device__ void range_in_place(Vec<2, T> minimum, Vec<2, T> maximum);

    __host__ __device__ Vec<3, T> to_3d(T z) const;

    __host__ __device__ explicit Vec(Polar<T> polar);
    __host__ __device__ Polar<T> to_polar() const;
};

template<typename T>
class Vec<3, T> {
public:
    T x;
    T y;
    T z;

    __host__ __device__ Vec();
    __host__ __device__ Vec(T x, T y, T z);
    __host__ __device__ Vec(const Vec<3, T>& other);
    __host__ __device__ ~Vec();

    __host__ __device__ Vec<3, T> operator+(Vec<3, T> other_vector) const;
    __host__ __device__ Vec<3, T> operator+(T other) const;
    __host__ __device__ void operator+=(Vec<3, T> other_vector);
    __host__ __device__ void operator+=(T other);
    __host__ __device__ Vec<3, T> add(Vec<3, T> other_vector) const;
    __host__ __device__ Vec<3, T> add(T other) const;
    __host__ __device__ void add_in_place(Vec<3, T> other_vector);
    __host__ __device__ void add_in_place(T other);

    __host__ __device__ Vec<3, T> operator-(Vec<3, T> other_vector) const;
    __host__ __device__ Vec<3, T> operator-(T other) const;
    __host__ __device__ void operator-=(Vec<3, T> other_vector);
    __host__ __device__ void operator-=(T other);
    __host__ __device__ Vec<3, T> subtract(Vec<3, T> other_vector) const;
    __host__ __device__ Vec<3, T> subtract(T other) const;
    __host__ __device__ void subtract_in_place(Vec<3, T> other_vector);
    __host__ __device__ void subtract_in_place(T other);

    __host__ __device__ Vec<3, T> operator*(Vec<3, T> other_vector) const;
    __host__ __device__ Vec<3, T> operator*(T other) const;
    __host__ __device__ void operator*=(Vec<3, T> other_vector);
    __host__ __device__ void operator*=(T other);
    __host__ __device__ Vec<3, T> multiply(Vec<3, T> other_vector) const;
    __host__ __device__ Vec<3, T> multiply(T other) const;
    __host__ __device__ void multiply_in_place(Vec<3, T> other_vector);
    __host__ __device__ void multiply_in_place(T other);

    __host__ __device__ Vec<3, T> operator/(Vec<3, T> other_vector) const;
    __host__ __device__ Vec<3, T> operator/(T other) const;
    __host__ __device__ void operator/=(Vec<3, T> other_vector);
    __host__ __device__ void operator/=(T other);
    __host__ __device__ Vec<3, T> divide(Vec<3, T> other_vector) const;
    __host__ __device__ Vec<3, T> divide(T other) const;
    __host__ __device__ void divide_in_place(Vec<3, T> other_vector);
    __host__ __device__ void divide_in_place(T other);

    __host__ __device__ Vec<3, T> power(Vec<3, T> other_vector) const;
    __host__ __device__ Vec<3, T> power(T other) const;
    __host__ __device__ void power_in_place(Vec<3, T> other_vector);
    __host__ __device__ void power_in_place(T other);

    __host__ __device__ Vec<3, T> abs() const;
    __host__ __device__ void abs_in_place();

    __host__ __device__ Vec<3, T> sqrt() const;
    __host__ __device__ void sqrt_in_place();

    __host__ __device__ Vec<3, T> round() const;
    __host__ __device__ void round_in_place();

    __host__ __device__ T get_distance_to(Vec<3, T> other_vector) const;
    __host__ __device__ T magnitude() const;

    __host__ __device__ Vec<3, T> normalize() const;
    __host__ __device__ void normalize_in_place();

    __host__ __device__ Vec<3, T> min(T minimum) const;
    __host__ __device__ Vec<3, T> min(Vec<3, T> minimum) const;
    __host__ __device__ Vec<3, T> max(T maximum) const;
    __host__ __device__ Vec<3, T> max(Vec<3, T> maximum) const;

    __host__ __device__ void min_in_place(T minimum);
    __host__ __device__ void min_in_place(Vec<3, T> minimum);
    __host__ __device__ void max_in_place(T maximum);
    __host__ __device__ void max_in_place(Vec<3, T> maximum);

    __host__ __device__ Vec<3, T> range(T minimum, T maximum) const;
    __host__ __device__ Vec<3, T> range(Vec<3, T> minimum, Vec<3, T> maximum) const;
    __host__ __device__ void range_in_place(T minimum, T maximum);
    __host__ __device__ void range_in_place(Vec<3, T> minimum, Vec<3, T> maximum);
};


template class Vec<2, float>;
template class Vec<3, float>;
template class Vec<2, double>;
template class Vec<3, double>;

#endif //VECTORLIB_VECTOR_CUH
