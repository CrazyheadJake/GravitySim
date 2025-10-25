#include "raylib.h"
#include "Planet.h"
#include "Simulation.cuh"
// #include "CudaHelpers.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <time.h>

float randf(float min, float max) {
    return ((float)rand() / RAND_MAX) * (max - min) + min;
}


int main() {
    // Initialize the window
    InitWindow(1280, 720, "Gravity Simulator");
    SetTargetFPS(144);

    srand(time(NULL));
    // Set up the simulation by initializing planets
    Simulation sim;
    for (int i = 0; i < 128; i++) {
        sim.addPlanet({randf(-100000, 100000), randf(-50000, 50000), 0}, {randf(-20, 20), randf(-20, 20), 0}, randf(1000000, 1000000));
    }
    sim.runSimulation();

    Vector2 center = {GetScreenWidth() / 2.0f, GetScreenHeight() / 2.0f};
    int metersPerPixel = 200;
    while (!WindowShouldClose()) {
        // Copies data from GPU to CPU once per frame
        sim.getPlanetsFromGPU();

        // Drawing planets to screen
        BeginDrawing();
        ClearBackground(WHITE);
        for (int i = 0; i < sim.getNumPlanets(); i++) {
            const Planet& p = sim.getPlanet(i);
            DrawCircle(center.x + p.pos.x / metersPerPixel, center.y + p.pos.y / metersPerPixel, std::max(sqrt(p.mass) / (double)metersPerPixel, 3.0), RED);
        }
        EndDrawing();
    }

    CloseWindow();
    exit(0);    // Must use exit() to force kernel to close
}