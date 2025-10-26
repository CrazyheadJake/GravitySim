#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

using vec3 = float3;

struct Planet {
    vec3 pos;   // meters
    vec3 vel;   // meters/sec
    float mass; // kilograms
};