#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include "nbody_kernel.cuh"

void checkCuda(cudaError_t result, char const *const func, const char *const file, int const line) {
  if (result) {
    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
    file << ":" << line << " '" << func << "' \n";
    cudaDeviceReset();
    exit(99);
  }
}
#define checkCudaErrors(val) checkCuda((val), #val, __FILE__, __LINE__)

void randomizeBodies(std::vector<Body>& bodies, int n) {
  std::mt19937 gen(1234);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (int i = 0; i < n; i++) {
    bodies[i].x = dist(gen);
    bodies[i].y = dist(gen);
    bodies[i].z = dist(gen);
    bodies[i].w = std::abs(dist(gen)) * 100.0f + 10.0f; // Mass
    bodies[i].vx = 0.0f;
    bodies[i].vy = 0.0f;
    bodies[i].vz = 0.0f;
  }
}

int main(int argc, char** argv) {
  int nBodies = 4096; // Default size
  if (argc > 1) nBodies = atoi(argv[1]);
  
  int nSteps = 50; // Simulation steps
  float dt = 0.01f;

  size_t bytes = nBodies * sizeof(Body);
  std::vector<Body> h_bodies(nBodies);
  randomizeBodies(h_bodies, nBodies);

  Body *d_bodies;
  checkCudaErrors(cudaMalloc(&d_bodies, bytes));
  checkCudaErrors(cudaMemcpy(d_bodies, h_bodies.data(), bytes, cudaMemcpyHostToDevice));

  int blockSize = BLOCK_SIZE;
  int gridSize = (nBodies + blockSize - 1) / blockSize;

  std::cout << "Running N-Body Simulation..." << std::endl;
  std::cout << "Bodies: " << nBodies << " | Steps: " << nSteps << std::endl;

  // Open CSV file for output
  std::ofstream outFile("nbody_output.csv");
  outFile << "step,id,x,y,z\n";

  // Simulation Loop
  for (int iter = 0; iter < nSteps; iter++) {
    integrateBodies<<<gridSize, blockSize>>>(d_bodies, dt, nBodies);
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Copy back to host to save (in a real high-perf app, we wouldn't copy back every frame)
    checkCudaErrors(cudaMemcpy(h_bodies.data(), d_bodies, bytes, cudaMemcpyDeviceToHost));

    // Save every 5th frame to keep file size small
    if (iter % 5 == 0) {
        for (int i = 0; i < nBodies; i+=10) { // Subsample bodies for plotting
             outFile << iter << "," << i << "," << h_bodies[i].x << "," << h_bodies[i].y << "," << h_bodies[i].z << "\n";
        }
    }
  }

  outFile.close();
  std::cout << "Simulation Complete. Data saved to nbody_output.csv" << std::endl;

  checkCudaErrors(cudaFree(d_bodies));
  return 0;
}