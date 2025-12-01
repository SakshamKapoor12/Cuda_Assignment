#ifndef NBODY_KERNEL_CUH
#define NBODY_KERNEL_CUH

#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256
#define SOFTENING 1e-9f

// Structure to represent a body
struct Body {
  float x, y, z, w; // Position x,y,z and Mass w
  float vx, vy, vz; // Velocity
};

// Device function to calculate interaction
__device__ void bodyBodyInteraction(float4 &myPos, float4 &otherPos, float3 &accel) {
  float dx = otherPos.x - myPos.x;
  float dy = otherPos.y - myPos.y;
  float dz = otherPos.z - myPos.z;
  float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
  float invDist = rsqrtf(distSqr);
  float invDist3 = invDist * invDist * invDist;
  
  float f = otherPos.w * invDist3;
  accel.x += dx * f;
  accel.y += dy * f;
  accel.z += dz * f;
}

// Tiled calculation kernel using Shared Memory
__global__ void integrateBodies(Body *p, float dt, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  float4 myPos;
  if (i < n) {
      myPos = make_float4(p[i].x, p[i].y, p[i].z, p[i].w);
  }

  float3 acc = {0.0f, 0.0f, 0.0f};

  // Shared memory cache for the "tile" of bodies
  __shared__ float4 sharedPos[BLOCK_SIZE];

  // Loop over all tiles of the grid
  for (int tile = 0; tile < gridDim.x; tile++) {
    int idx = tile * blockDim.x + threadIdx.x;
    
    // 1. Load data into shared memory
    if (idx < n) {
       sharedPos[threadIdx.x] = make_float4(p[idx].x, p[idx].y, p[idx].z, p[idx].w);
    } else {
       sharedPos[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    
    __syncthreads(); // Barrier to ensure tile is loaded

    // 2. Compute force with all bodies in this tile
    // This reduces global memory bandwidth pressure significantly
    if (i < n) {
        #pragma unroll
        for (int j = 0; j < BLOCK_SIZE; j++) {
            bodyBodyInteraction(myPos, sharedPos[j], acc);
        }
    }
    
    __syncthreads(); // Barrier before loading next tile
  }

  // 3. Update velocity and position
  if (i < n) {
    p[i].vx += acc.x * dt;
    p[i].vy += acc.y * dt;
    p[i].vz += acc.z * dt;

    p[i].x += p[i].vx * dt;
    p[i].y += p[i].vy * dt;
    p[i].z += p[i].vz * dt;
  }
}

#endif