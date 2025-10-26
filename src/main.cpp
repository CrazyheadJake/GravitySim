#include "raylib.h"
#include "Planet.h"
#include "Simulation.cuh"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <time.h>
#include <string>
#include <chrono>

float randf(float min, float max) {
    return ((float)rand() / RAND_MAX) * (max - min) + min;
}

using clk = std::chrono::steady_clock;

int main() {
    // Initialize the window
    int width = 1280;
    int height = 720;
    int64_t metersPerPixel = 6000000;

    InitWindow(1280, 720, "Gravity Simulator");
    SetTargetFPS(144);

    // Set up the simulation by initializing planets
    srand(time(NULL));
    Simulation sim;
    sim.addPlanet({0, 0, 0}, {0, 0, 0}, 1.989e30);
    for (int i = 0; i < 1024; i++) {
        sim.addPlanet({randf(-metersPerPixel * width/2, metersPerPixel * width/2), randf(-metersPerPixel * height/2, metersPerPixel * height/2), 0},
         {randf(0, 0), randf(-metersPerPixel / 10, 0), 0}, 
         randf(1, 6e25));
    }
    sim.runSimulation();

    Vector2 center = {GetScreenWidth() / 2.0f, GetScreenHeight() / 2.0f};
    int lastSimulationCount = 0;
    int simCounter = 0;
    int frameCounter = 0;
    std::string simRateTxt;
    std::string fps;
    clk::time_point last = clk::now();

    while (!WindowShouldClose()) {
        // Copies data from GPU to CPU once per frame
        sim.getPlanetsFromGPU();

        // Get scrolling input
        float scroll = GetMouseWheelMove();
        if (scroll > 0)
            metersPerPixel /= 1.1;
        else if (scroll < 0) {
            metersPerPixel *= 1.1;
            std::cout << metersPerPixel << std::endl;
        }

        // Drawing planets to screen
        BeginDrawing();
        ClearBackground(WHITE);
        for (int i = 0; i < sim.getNumPlanets(); i++) {
            const Planet& p = sim.getPlanet(i);
            DrawCircle(center.x + p.pos.x / metersPerPixel, center.y + p.pos.y / metersPerPixel, std::max(cbrt(p.mass / 6e24) * 6000000 / metersPerPixel , 3.0), RED);
        }

        // Frame rate and Simulation rate info
        if (frameCounter % 72 == 0) {
            lastSimulationCount = simCounter;
            simCounter = sim.getFrameCountFromGPU();
            clk::time_point now = clk::now();
            float dt = std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count() / 1000.0f;
            last = now;
            simRateTxt = std::string("Simulation Rate: ").append(std::to_string((int)((simCounter - lastSimulationCount) / dt)));
            fps = std::string("Frame Rate: ").append(std::to_string(GetFPS()));
            frameCounter = 0;
        }
        else {
            GetFPS();   // Must call this every frame so that it is accurate
        }
        frameCounter++;
        DrawText(simRateTxt.c_str(), 10, 10, 20, BLACK);
        DrawText(fps.c_str(), 10, 30, 20, BLACK);

        EndDrawing();
    }

    CloseWindow();
    exit(0);    // Must use exit() to force kernel to close
}