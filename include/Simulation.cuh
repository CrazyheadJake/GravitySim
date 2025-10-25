#pragma once
#include "Planet.h"
#include <cuda_runtime.h>   // CUDA runtime API
#include <vector>

class Simulation {
public:
    Simulation();
    ~Simulation();

    void runSimulation();
    void getPlanetsFromGPU();
    const Planet& getPlanet(int i) { return m_hPlanets[i]; }
    int getNumPlanets() { return m_numPlanets; }
    void addPlanet(float3 pos, float3 vel, float mass);

private:
    std::vector<Planet> m_planets;
    int m_numPlanets;
    Planet* m_hPlanets;
    Planet* m_dPlanets;
    Planet* m_dNextPlanets;
    void** m_args;
    cudaStream_t m_kernelStream, m_dataStream;

};