#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

struct Planet {
    float3 pos;
    float3 vel;
    float mass;
};