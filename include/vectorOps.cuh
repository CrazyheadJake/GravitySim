__host__ __device__ inline vec3 operator-(const vec3& a, const vec3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline vec3 operator+(const vec3& a, const vec3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline vec3 operator*(const vec3& a, double s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ inline vec3 operator*(float s, const vec3& a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ inline vec3 operator+=(vec3& a, const vec3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

__host__ __device__ inline vec3& operator*=(vec3& a, double b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    return a;
}

__host__ __device__ inline float dot(const vec3& a, const vec3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}