#include <cmath>

#include "Vector.cuh"
#include "Polar.cuh"


template<typename T>
__host__ __device__ Vec<2, T>::Vec() {}

template<typename T>
__host__ __device__ Vec<2, T>::Vec(const T x, const T y): x(x), y(y) {}

template<typename T>
__host__ __device__ Vec<2, T>::Vec(const Vec<2, T>& other): x(other.x), y(other.y) {}

template<typename T>
__host__ __device__ Vec<2, T>::~Vec() {}

template<typename T>
__host__ __device__ Vec<2, T> Vec<2, T>::operator+(const Vec<2, T> other_vector) const {
    return {this->x + other_vector.x, this->y + other_vector.y};
}

template<typename T>
__host__ __device__ Vec<2, T> Vec<2, T>::operator+(const T other) const {
    return {this->x + other, this->y + other};
}

template<typename T>
__host__ __device__ void Vec<2, T>::operator+=(const Vec<2, T> other_vector) {
    this->x += other_vector.x;
    this->y += other_vector.y;
}

template<typename T>
__host__ __device__ void Vec<2, T>::operator+=(const T other) {
    this->x += other;
    this->y += other;
}

template<typename T>
__host__ __device__ Vec<2, T> Vec<2, T>::add(const Vec<2, T> other_vector) const {
    return {this->x + other_vector.x, this->y + other_vector.y};
}

template<typename T>
__host__ __device__ Vec<2, T> Vec<2, T>::add(const T other) const {
    return {this->x + other, this->y + other};
}

template<typename T>
__host__ __device__ void Vec<2, T>::add_in_place(const Vec<2, T> other_vector) {
    this->x += other_vector.x;
    this->y += other_vector.y;
}

template<typename T>
__host__ __device__ void Vec<2, T>::add_in_place(const T other) {
    this->x += other;
    this->y += other;
}

template<typename T>
__host__ __device__ Vec<2, T> Vec<2, T>::operator-(const Vec<2, T> other_vector) const {
    return {this->x - other_vector.x, this->y - other_vector.y};
}

template<typename T>
__host__ __device__ Vec<2, T> Vec<2, T>::operator-(const T other) const {
    return {this->x - other, this->y - other};
}

template<typename T>
__host__ __device__ void Vec<2, T>::operator-=(const Vec<2, T> other_vector) {
    this->x -= other_vector.x;
    this->y -= other_vector.y;
}

template<typename T>
__host__ __device__ void Vec<2, T>::operator-=(const T other) {
    this->x -= other;
    this->y -= other;
}

template<typename T>
__host__ __device__ Vec<2, T> Vec<2, T>::subtract(const Vec<2, T> other_vector) const {
    return {this->x - other_vector.x, this->y - other_vector.y};
}

template<typename T>
__host__ __device__ Vec<2, T> Vec<2, T>::subtract(const T other) const {
    return {this->x - other, this->y - other};
}

template<typename T>
__host__ __device__ void Vec<2, T>::subtract_in_place(const Vec<2, T> other_vector) {
    this->x -= other_vector.x;
    this->y -= other_vector.y;
}

template<typename T>
__host__ __device__ void Vec<2, T>::subtract_in_place(const T other) {
    this->x -= other;
    this->y -= other;
}

template<typename T>
__host__ __device__ Vec<2, T> Vec<2, T>::operator*(const Vec<2, T> other_vector) const {
    return {this->x * other_vector.x, this->y * other_vector.y};
}

template<typename T>
__host__ __device__ Vec<2, T> Vec<2, T>::operator*(const T other) const {
    return {this->x * other, this->y * other};
}

template<typename T>
__host__ __device__ void Vec<2, T>::operator*=(const Vec<2, T> other_vector) {
    this->x *= other_vector.x;
    this->y *= other_vector.y;
}

template<typename T>
__host__ __device__ void Vec<2, T>::operator*=(const T other) {
    this->x *= other;
    this->y *= other;
}

template<typename T>
__host__ __device__ Vec<2, T> Vec<2, T>::multiply(const Vec<2, T> other_vector) const {
    return {this->x * other_vector.x, this->y * other_vector.y};
}

template<typename T>
__host__ __device__ Vec<2, T> Vec<2, T>::multiply(const T other) const {
    return {this->x * other, this->y * other};
}

template<typename T>
__host__ __device__ void Vec<2, T>::multiply_in_place(const Vec<2, T> other_vector) {
    this->x *= other_vector.x;
    this->y *= other_vector.y;
}

template<typename T>
__host__ __device__ void Vec<2, T>::multiply_in_place(const T other) {
    this->x *= other;
    this->y *= other;
}

template<typename T>
__host__ __device__ Vec<2, T> Vec<2, T>::operator/(const Vec<2, T> other_vector) const {
    return {this->x / other_vector.x, this->y / other_vector.y};
}

template<typename T>
__host__ __device__ Vec<2, T> Vec<2, T>::operator/(const T other) const {
    return {this->x / other, this->y / other};
}

template<typename T>
__host__ __device__ void Vec<2, T>::operator/=(const Vec<2, T> other_vector) {
    this->x /= other_vector.x;
    this->y /= other_vector.y;
}

template<typename T>
__host__ __device__ void Vec<2, T>::operator/=(const T other) {
    this->x /= other;
    this->y /= other;
}

template<typename T>
__host__ __device__ Vec<2, T> Vec<2, T>::divide(const Vec<2, T> other_vector) const {
    return {this->x / other_vector.x, this->y / other_vector.y};
}

template<typename T>
__host__ __device__ Vec<2, T> Vec<2, T>::divide(const T other) const {
    return {this->x / other, this->y / other};
}

template<typename T>
__host__ __device__ void Vec<2, T>::divide_in_place(const Vec<2, T> other_vector) {
    this->x /= other_vector.x;
    this->y /= other_vector.y;
}

template<typename T>
__host__ __device__ void Vec<2, T>::divide_in_place(const T other) {
    this->x /= other;
    this->y /= other;
}

template<typename T>
__host__ __device__ Vec<2, T> Vec<2, T>::power(const Vec<2, T> other_vector) const {
    return {std::pow(this->x, other_vector.x), std::pow(this->y, other_vector.y)};
}

template<typename T>
__host__ __device__ Vec<2, T> Vec<2, T>::power(const T other) const {
    return {std::pow(this->x, other), std::pow(this->y, other)};
}

template<typename T>
__host__ __device__ void Vec<2, T>::power_in_place(const Vec<2, T> other_vector) {
    this->x = std::pow(this->x, other_vector.x);
    this->y = std::pow(this->y, other_vector.y);
}

template<typename T>
__host__ __device__ void Vec<2, T>::power_in_place(const T other) {
    this->x = std::pow(this->x, other);
    this->y = std::pow(this->y, other);
}

template<typename T>
__host__ __device__ Vec<2, T> Vec<2, T>::abs() const {
    return {std::abs(this->x), std::abs(this->y)};
}

template<typename T>
__host__ __device__ void Vec<2, T>::abs_in_place() {
    this->x = std::abs(x);
    this->y = std::abs(y);
}

template<typename T>
__host__ __device__ Vec<2, T> Vec<2, T>::sqrt() const {
    return {std::sqrt(this->x), std::sqrt(this->y)};
}

template<typename T>
__host__ __device__ void Vec<2, T>::sqrt_in_place() {
    this->x = std::sqrt(x);
    this->y = std::sqrt(y);
}

template<typename T>
__host__ __device__ Vec<2, T> Vec<2, T>::round() const {
    return {std::round(this->x), std::round(this->y)};
}

template<typename T>
__host__ __device__ void Vec<2, T>::round_in_place() {
    this->x = std::round(x);
    this->y = std::round(y);
}

template<typename T>
__host__ __device__ T Vec<2, T>::get_distance_to(Vec<2, T> other_vector) const {
    return std::sqrt(std::pow(this->x - other_vector.x, 2) + std::pow(this->y - other_vector.y, 2));
}

template<typename T>
__host__ __device__ T Vec<2, T>::magnitude() const {
    return std::sqrt(this->x * this->x + this->y * this->y);
}

template<typename T>
__host__ __device__ T Vec<2, T>::angle() const {
    return std::atan2(this->y, this->x);
}

template<typename T>
__host__ __device__ Vec<2, T> Vec<2, T>::normalize() const {
    return this->divide(this->magnitude());
}

template<typename T>
__host__ __device__ void Vec<2, T>::normalize_in_place() {
    this->divide_in_place(this->magnitude());
}

template<typename T>
__host__ __device__ Vec<2, T> Vec<2, T>::rotate(const T angle) const {
    return {std::cos(angle) * this->x - std::sin(angle) * this->y, std::sin(angle) * this->x + std::cos(angle) * this->y};
}

template<typename T>
__host__ __device__ void Vec<2, T>::rotate_in_place(const T angle) {
    this->x = std::cos(angle) * this->x - std::sin(angle) * this->y;
    this->y = std::sin(angle) * this->x + std::cos(angle) * this->y;
}

template<typename T>
__host__ __device__ Vec<2, T> Vec<2, T>::min(const T minimum) const {
    return {std::min(this->x, minimum), std::min(this->y, minimum)};
}

template<typename T>
__host__ __device__ Vec<2, T> Vec<2, T>::min(const Vec<2, T> minimum) const {
    return {std::min(this->x, minimum.x), std::min(this->y, minimum.y)};
}

template<typename T>
__host__ __device__ Vec<2, T> Vec<2, T>::max(const T maximum) const {
    return {std::max(this->x, maximum), std::max(this->y, maximum)};
}

template<typename T>
__host__ __device__ Vec<2, T> Vec<2, T>::max(const Vec<2, T> maximum) const {
    return {std::max(this->x, maximum.x), std::max(this->y, maximum.y)};
}

template<typename T>
__host__ __device__ void Vec<2, T>::min_in_place(const T minimum) {
    this->x = std::min(this->x, minimum);
    this->y = std::min(this->y, minimum);
}

template<typename T>
__host__ __device__ void Vec<2, T>::min_in_place(const Vec<2, T> minimum) {
    this->x = std::min(this->x, minimum.x);
    this->y = std::min(this->y, minimum.y);
}

template<typename T>
__host__ __device__ void Vec<2, T>::max_in_place(const T maximum) {
    this->x = std::max(this->x, maximum);
    this->y = std::max(this->y, maximum);
}

template<typename T>
__host__ __device__ void Vec<2, T>::max_in_place(const Vec<2, T> maximum) {
    this->x = std::max(this->x, maximum.x);
    this->y = std::max(this->y, maximum.y);
}

template<typename T>
__host__ __device__ Vec<2, T> Vec<2, T>::range(const T minimum, const T maximum) const {
    return {std::max(std::min(this->x, minimum), maximum), std::max(std::min(this->y, minimum), maximum)};
}

template<typename T>
__host__ __device__ Vec<2, T> Vec<2, T>::range(const Vec<2, T> minimum, const Vec<2, T> maximum) const {
    return {std::max(std::min(this->x, minimum.x), maximum.x), std::max(std::min(this->y, minimum.y), maximum.y)};
}

template<typename T>
__host__ __device__ void Vec<2, T>::range_in_place(const T minimum, const T maximum) {
    this->x = std::max(std::min(this->x, minimum), maximum);
    this->y = std::max(std::min(this->y, minimum), maximum);
}

template<typename T>
__host__ __device__ void Vec<2, T>::range_in_place(const Vec<2, T> minimum, const Vec<2, T> maximum) {
    this->x = std::max(std::min(this->x, minimum.x), maximum.x);
    this->y = std::max(std::min(this->y, minimum.y), maximum.y);
}

template<typename T>
__host__ __device__ Vec<3, T> Vec<2, T>::to_3d(const T z) const {
    return {this->x, this->y, z};
}

template<typename T>
__host__ __device__ Vec<2, T>::Vec(const Polar<T> polar): x(polar.magnitude * std::cos(polar.angle)), y(polar.magnitude * std::sin(polar.angle)) {}

template<typename T>
__host__ __device__ Polar<T> Vec<2, T>::to_polar() const {
    return {this->magnitude(), this->angle()};
}


template<typename T>
__host__ __device__ Vec<3, T>::Vec() {}

template<typename T>
__host__ __device__ Vec<3, T>::Vec(const T x, const T y, const T z): x(x), y(y), z(z) {}

template<typename T>
__host__ __device__ Vec<3, T>::Vec(const Vec<3, T>& other): x(other.x), y(other.y), z(other.z) {}

template<typename T>
__host__ __device__ Vec<3, T>::~Vec() {}

template<typename T>
__host__ __device__ Vec<3, T> Vec<3, T>::operator+(const Vec<3, T> other_vector) const {
    return {this->x + other_vector.x, this->y + other_vector.y, this->z + other_vector.z};
}

template<typename T>
__host__ __device__ Vec<3, T> Vec<3, T>::operator+(const T other) const {
    return {this->x + other, this->y + other, this->z + other};
}

template<typename T>
__host__ __device__ void Vec<3, T>::operator+=(const Vec<3, T> other_vector) {
    this->x += other_vector.x;
    this->y += other_vector.y;
    this->z += other_vector.z;
}

template<typename T>
__host__ __device__ void Vec<3, T>::operator+=(const T other) {
    this->x += other;
    this->y += other;
    this->z += other;
}

template<typename T>
__host__ __device__ Vec<3, T> Vec<3, T>::add(const Vec<3, T> other_vector) const {
    return {this->x + other_vector.x, this->y + other_vector.y, this->z + other_vector.z};
}

template<typename T>
__host__ __device__ Vec<3, T> Vec<3, T>::add(const T other) const {
    return {this->x + other, this->y + other, this->z + other};
}

template<typename T>
__host__ __device__ void Vec<3, T>::add_in_place(const Vec<3, T> other_vector) {
    this->x += other_vector.x;
    this->y += other_vector.y;
    this->z += other_vector.z;
}

template<typename T>
__host__ __device__ void Vec<3, T>::add_in_place(const T other) {
    this->x += other;
    this->y += other;
    this->z += other;
}

template<typename T>
__host__ __device__ Vec<3, T> Vec<3, T>::operator-(const Vec<3, T> other_vector) const {
    return {this->x - other_vector.x, this->y - other_vector.y, this->z - other_vector.z};
}

template<typename T>
__host__ __device__ Vec<3, T> Vec<3, T>::operator-(const T other) const {
    return {this->x - other, this->y - other, this->z - other};
}

template<typename T>
__host__ __device__ void Vec<3, T>::operator-=(const Vec<3, T> other_vector) {
    this->x -= other_vector.x;
    this->y -= other_vector.y;
    this->z -= other_vector.z;
}

template<typename T>
__host__ __device__ void Vec<3, T>::operator-=(const T other) {
    this->x -= other;
    this->y -= other;
    this->z -= other;
}

template<typename T>
__host__ __device__ Vec<3, T> Vec<3, T>::subtract(const Vec<3, T> other_vector) const {
    return {this->x - other_vector.x, this->y - other_vector.y, this->z - other_vector.z};
}

template<typename T>
__host__ __device__ Vec<3, T> Vec<3, T>::subtract(const T other) const {
    return {this->x - other, this->y - other, this->z - other};
}

template<typename T>
__host__ __device__ void Vec<3, T>::subtract_in_place(const Vec<3, T> other_vector) {
    this->x -= other_vector.x;
    this->y -= other_vector.y;
    this->z -= other_vector.z;
}

template<typename T>
__host__ __device__ void Vec<3, T>::subtract_in_place(const T other) {
    this->x -= other;
    this->y -= other;
    this->z -= other;
}

template<typename T>
__host__ __device__ Vec<3, T> Vec<3, T>::operator*(const Vec<3, T> other_vector) const {
    return {this->x * other_vector.x, this->y * other_vector.y, this->z * other_vector.z};
}

template<typename T>
__host__ __device__ Vec<3, T> Vec<3, T>::operator*(const T other) const {
    return {this->x * other, this->y * other, this->z * other};
}

template<typename T>
__host__ __device__ void Vec<3, T>::operator*=(const Vec<3, T> other_vector) {
    this->x *= other_vector.x;
    this->y *= other_vector.y;
    this->z *= other_vector.z;
}

template<typename T>
__host__ __device__ void Vec<3, T>::operator*=(const T other) {
    this->x *= other;
    this->y *= other;
    this->z *= other;
}

template<typename T>
__host__ __device__ Vec<3, T> Vec<3, T>::multiply(const Vec<3, T> other_vector) const {
    return {this->x * other_vector.x, this->y * other_vector.y, this->z * other_vector.z};
}

template<typename T>
__host__ __device__ Vec<3, T> Vec<3, T>::multiply(const T other) const {
    return {this->x * other, this->y * other, this->z * other};
}

template<typename T>
__host__ __device__ void Vec<3, T>::multiply_in_place(const Vec<3, T> other_vector) {
    this->x *= other_vector.x;
    this->y *= other_vector.y;
    this->z *= other_vector.z;
}

template<typename T>
__host__ __device__ void Vec<3, T>::multiply_in_place(const T other) {
    this->x *= other;
    this->y *= other;
    this->z *= other;
}

template<typename T>
__host__ __device__ Vec<3, T> Vec<3, T>::operator/(const Vec<3, T> other_vector) const {
    return {this->x / other_vector.x, this->y / other_vector.y, this->z / other_vector.z};
}

template<typename T>
__host__ __device__ Vec<3, T> Vec<3, T>::operator/(const T other) const {
    return {this->x / other, this->y / other, this->z / other};
}

template<typename T>
__host__ __device__ void Vec<3, T>::operator/=(const Vec<3, T> other_vector) {
    this->x /= other_vector.x;
    this->y /= other_vector.y;
    this->z /= other_vector.z;
}

template<typename T>
__host__ __device__ void Vec<3, T>::operator/=(const T other) {
    this->x /= other;
    this->y /= other;
    this->z /= other;
}

template<typename T>
__host__ __device__ Vec<3, T> Vec<3, T>::divide(const Vec<3, T> other_vector) const {
    return {this->x / other_vector.x, this->y / other_vector.y, this->z / other_vector.z};
}

template<typename T>
__host__ __device__ Vec<3, T> Vec<3, T>::divide(const T other) const {
    return {this->x / other, this->y / other, this->z / other};
}

template<typename T>
__host__ __device__ void Vec<3, T>::divide_in_place(const Vec<3, T> other_vector) {
    this->x /= other_vector.x;
    this->y /= other_vector.y;
    this->z /= other_vector.z;
}

template<typename T>
__host__ __device__ void Vec<3, T>::divide_in_place(const T other) {
    this->x /= other;
    this->y /= other;
    this->z /= other;
}

template<typename T>
__host__ __device__ Vec<3, T> Vec<3, T>::power(const Vec<3, T> other_vector) const {
    return {std::pow(this->x, other_vector.x), std::pow(this->y, other_vector.y), std::pow(this->z, other_vector.z)};
}

template<typename T>
__host__ __device__ Vec<3, T> Vec<3, T>::power(const T other) const {
    return {std::pow(this->x, other), std::pow(this->y, other), std::pow(this->z, other)};
}

template<typename T>
__host__ __device__ void Vec<3, T>::power_in_place(const Vec<3, T> other_vector) {
    this->x = std::pow(this->x, other_vector.x);
    this->y = std::pow(this->y, other_vector.y);
    this->z = std::pow(this->z, other_vector.z);
}

template<typename T>
__host__ __device__ void Vec<3, T>::power_in_place(const T other) {
    this->x = std::pow(this->x, other);
    this->y = std::pow(this->y, other);
    this->z = std::pow(this->z, other);
}

template<typename T>
__host__ __device__ Vec<3, T> Vec<3, T>::abs() const {
    return {std::abs(this->x), std::abs(this->y), std::abs(this->z)};
}

template<typename T>
__host__ __device__ void Vec<3, T>::abs_in_place() {
    this->x = std::abs(x);
    this->y = std::abs(y);
    this->z = std::abs(z);
}

template<typename T>
__host__ __device__ Vec<3, T> Vec<3, T>::sqrt() const {
    return {std::sqrt(this->x), std::sqrt(this->y), std::sqrt(this->z)};
}

template<typename T>
__host__ __device__ void Vec<3, T>::sqrt_in_place() {
    this->x = std::sqrt(x);
    this->y = std::sqrt(y);
    this->z = std::sqrt(z);
}

template<typename T>
__host__ __device__ Vec<3, T> Vec<3, T>::round() const {
    return {std::round(this->x), std::round(this->y), std::round(this->z)};
}

template<typename T>
__host__ __device__ void Vec<3, T>::round_in_place() {
    this->x = std::round(x);
    this->y = std::round(y);
    this->z = std::round(z);
}

template<typename T>
__host__ __device__ T Vec<3, T>::get_distance_to(Vec<3, T> other_vector) const {
    return std::sqrt(std::pow(this->x - other_vector.x, 2) + std::pow(this->y - other_vector.y, 2)  + std::pow(this->z - other_vector.z, 2));
}

template<typename T>
__host__ __device__ T Vec<3, T>::magnitude() const {
    return std::sqrt(this->x * this->x + this->y * this->y + this->z * this->z);
}

template<typename T>
__host__ __device__ Vec<3, T> Vec<3, T>::normalize() const {
    return this->divide(this->magnitude());
}

template<typename T>
__host__ __device__ void Vec<3, T>::normalize_in_place() {
    this->divide_in_place(this->magnitude());
}

template<typename T>
__host__ __device__ Vec<3, T> Vec<3, T>::min(const T minimum) const {
    return {std::min(this->x, minimum), std::min(this->y, minimum), std::min(this->z, minimum)};
}

template<typename T>
__host__ __device__ Vec<3, T> Vec<3, T>::min(const Vec<3, T> minimum) const {
    return {std::min(this->x, minimum.x), std::min(this->y, minimum.y), std::min(this->z, minimum.z)};
}

template<typename T>
__host__ __device__ Vec<3, T> Vec<3, T>::max(const T maximum) const {
    return {std::max(this->x, maximum), std::max(this->y, maximum), std::max(this->z, maximum)};
}

template<typename T>
__host__ __device__ Vec<3, T> Vec<3, T>::max(const Vec<3, T> maximum) const {
    return {std::max(this->x, maximum.x), std::max(this->y, maximum.y), std::max(this->z, maximum.z)};
}

template<typename T>
__host__ __device__ void Vec<3, T>::min_in_place(const T minimum) {
    this->x = std::min(this->x, minimum);
    this->y = std::min(this->y, minimum);
    this->z = std::min(this->z, minimum);
}

template<typename T>
__host__ __device__ void Vec<3, T>::min_in_place(const Vec<3, T> minimum) {
    this->x = std::min(this->x, minimum.x);
    this->y = std::min(this->y, minimum.y);
    this->z = std::min(this->z, minimum.z);
}

template<typename T>
__host__ __device__ void Vec<3, T>::max_in_place(const T maximum) {
    this->x = std::max(this->x, maximum);
    this->y = std::max(this->y, maximum);
}

template<typename T>
__host__ __device__ void Vec<3, T>::max_in_place(const Vec<3, T> maximum) {
    this->x = std::max(this->x, maximum.x);
    this->y = std::max(this->y, maximum.y);
}

template<typename T>
__host__ __device__ Vec<3, T> Vec<3, T>::range(const T minimum, const T maximum) const {
    return {std::max(std::min(this->x, minimum), maximum), std::max(std::min(this->y, minimum), maximum), std::max(std::min(this->z, minimum), maximum)};
}

template<typename T>
__host__ __device__ Vec<3, T> Vec<3, T>::range(const Vec<3, T> minimum, const Vec<3, T> maximum) const {
    return {std::max(std::min(this->x, minimum.x), maximum.x), std::max(std::min(this->y, minimum.y), maximum.y), std::max(std::min(this->z, minimum.z), maximum.z)};
}

template<typename T>
__host__ __device__ void Vec<3, T>::range_in_place(const Vec<3, T> minimum, const Vec<3, T> maximum) {
    this->x = std::max(std::min(this->x, minimum.x), maximum.x);
    this->y = std::max(std::min(this->y, minimum.y), maximum.y);
    this->z = std::max(std::min(this->z, minimum.z), maximum.z);
}

template<typename T>
__host__ __device__ void Vec<3, T>::range_in_place(const T minimum, const T maximum) {
    this->x = std::max(std::min(this->x, minimum), maximum);
    this->y = std::max(std::min(this->y, minimum), maximum);
    this->z = std::max(std::min(this->z, minimum), maximum);
}
