#include <stdio.h>
#include <chrono>
#include <thread>
#include "2dParticleSimulation/include/renderer.h"
#include "2dParticleSimulation/include/utils.h"

#define TARGET_FRAME_TIME 1/60 // 60 fps


int main() {
    Renderer renderer;

    renderer.run();
    // using clock = std::chrono::high_resolution_clock;
    // double sleepTime;
    // while (!glfwWindowShouldClose(renderer.window)) {

    //     auto frameStart = clock::now();
        
    //     glfwPollEvents();

    //     // event start 
    //     renderer.run();

    //     // event end

    //     auto frameEnd = clock::now();

    //     std::chrono::duration<double> elapsed = frameEnd - frameStart;
    //     sleepTime = TARGET_FRAME_TIME - elapsed.count();
    //     if (sleepTime > 0.0f) {
    //         std::this_thread::sleep_for(std::chrono::duration<double>(sleepTime));
    //     }
    // }
    
    printf("\ntest okay\n\n");
}