#include <cuda_runtime.h>   // CUDA runtime API
#include <device_launch_parameters.h> // Optional: threadIdx, blockIdx, etc.
#include <cooperative_groups.h>
#include <iostream>
#include "Planet.h"
#include "Simulation.cuh"
#include <cuda/barrier>
#include "vectorOps.cuh"
#include "CudaHelpers.h"

namespace cg = cooperative_groups;

__device__ int frameCounter = 0;

__global__ void simulationKernel(Planet* dPlanets, Planet* dNextPlanets, int numPlanets) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    cg::grid_group grid = cg::this_grid();
    vec3 acc = {0, 0, 0};
    double G = 6.67430e-11; // N*m^2/kg^2
    float dt = 5;        // seconds

    // Return any threads that aren't being used
    if (idx >= numPlanets) {
        return;
    }

    while (true) {
        // Using Euler's method currently
        // Step 1: Calculate acceleration from all other planets' positions and masses
        acc = {0, 0, 0};    // meters/second^2
        for (int i = 0; i < numPlanets; i++) {
            if (i == idx)
                continue;
            vec3 diff = dPlanets[i].pos - dPlanets[idx].pos;
            float dist2 = dot(diff, diff);
            float inv_r = rsqrtf(dist2);
            acc += diff * (float)(G * dPlanets[i].mass) * inv_r * inv_r * inv_r;
        }
        dNextPlanets[idx].pos = dPlanets[idx].pos + dPlanets[idx].vel * dt;
        dNextPlanets[idx].vel = dPlanets[idx].vel + acc * dt;

        // Step 2: Synchronize all threads across the grid (each thread is 1 planet), and swap our two buffers 
        grid.sync();
        if (idx == 0)
            frameCounter += 1;

        cuda::std::swap(dPlanets, dNextPlanets);
    }
}



void Simulation::getPlanetsFromGPU()
{
    // Must launch it as an async request from a separate stream to not block and wait for the kernel to finish
    cudaMemcpyAsync(m_hPlanets, m_dPlanets, m_numPlanets * sizeof(Planet), cudaMemcpyDeviceToHost, m_dataStream);
    cudaStreamSynchronize(m_dataStream);
    CudaHelpers::checkCudaErrors();
}

int Simulation::getFrameCountFromGPU()
{
    int hostValue;
    cudaMemcpyFromSymbolAsync(&hostValue, frameCounter, sizeof(int), 0, cudaMemcpyDeviceToHost, m_dataStream);
    cudaStreamSynchronize(m_dataStream);
    CudaHelpers::checkCudaErrors();
    return hostValue;
}

void Simulation::addPlanet(vec3 pos, vec3 vel, float mass)
{
    m_planets.push_back({pos, vel, mass});
}

Simulation::Simulation()
{
    // Initialize our arguments for the kernel
    m_args = (void **)malloc(sizeof(void*) * 3);
    m_args[0] = &m_hPlanets;
    m_args[1] = &m_dPlanets;
    m_args[2] = &m_numPlanets;
    
    cudaStreamCreate(&m_kernelStream);
    cudaStreamCreate(&m_dataStream);
    CudaHelpers::checkCudaErrors();
}

Simulation::~Simulation()
{
    free(m_args);
    cudaStreamDestroy(m_kernelStream);
    cudaStreamDestroy(m_dataStream);
    cudaFree(m_dPlanets);
    cudaFree(m_dNextPlanets);
    cudaFreeHost(m_hPlanets);
}

void Simulation::runSimulation()
{
    m_numPlanets = m_planets.size();

    const int threadsPerBlock = 128;
    const int blocks = m_numPlanets / (threadsPerBlock + 1) + 1;

    // Initialize variables on the host (CPU)
    cudaHostAlloc((void **)(&m_hPlanets), m_numPlanets * sizeof(Planet), cudaHostAllocDefault);
    CudaHelpers::checkCudaErrors();
    for (int i = 0; i < m_numPlanets; i++) {
        m_hPlanets[i] = m_planets[i];
    }
        
    // Initialize and allocate memory on the device (GPU)
    cudaMalloc((void **)(&m_dPlanets), m_numPlanets * sizeof(Planet));
    cudaMalloc((void **)(&m_dNextPlanets), m_numPlanets * sizeof(Planet));
    CudaHelpers::checkCudaErrors();

    // Copy values over to the device from the host
    cudaMemcpy(m_dPlanets, m_hPlanets, m_numPlanets * sizeof(Planet), cudaMemcpyHostToDevice);
    cudaMemcpy(m_dNextPlanets, m_hPlanets, m_numPlanets * sizeof(Planet), cudaMemcpyHostToDevice);
    CudaHelpers::checkCudaErrors();

    // Launch the kernel as cooperative so we can use grid.sync(), launch in a separate stream to be async
    std::cout << "Launching Kernel" << std::endl;
    cudaLaunchCooperativeKernel((void*)simulationKernel, blocks, threadsPerBlock, m_args, 0, m_kernelStream);
    std::cout << "After launch" << std::endl;
    CudaHelpers::checkCudaErrors();
}
