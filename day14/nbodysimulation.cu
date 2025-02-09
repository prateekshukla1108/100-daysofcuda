#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 128
#define G 1.0f
#define SOFTENING 1e-9f

struct Body {
    float3 pos;
    float3 vel;
    float mass;
};

__device__ inline float3 addFloat3(const float3 &a, const float3 &b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 subFloat3(const float3 &a, const float3 &b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 scaleFloat3(const float3 &a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__global__ void nbodyKernel(const Body *bodies, Body *new_bodies, int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    Body myBody = bodies[i];
    float3 acc = make_float3(0.0f, 0.0f, 0.0f);

    extern __shared__ Body sharedBodies[];
    int numTiles = (n + blockDim.x - 1) / blockDim.x;

    for (int tile = 0; tile < numTiles; tile++) {
        int idx = tile * blockDim.x + threadIdx.x;
        // Load body from global memory into shared memory tile
        if (idx < n)
            sharedBodies[threadIdx.x] = bodies[idx];
        else {
            sharedBodies[threadIdx.x].pos = make_float3(0.0f, 0.0f, 0.0f);
            sharedBodies[threadIdx.x].vel = make_float3(0.0f, 0.0f, 0.0f);
            sharedBodies[threadIdx.x].mass = 0.0f;
        }
        __syncthreads();

        // Check if the current tile contains our own body.
        if (tile == i / blockDim.x) {
            // Process bodies before our own index in the tile.
            for (int j = 0; j < threadIdx.x; j++) {
                float3 r = subFloat3(sharedBodies[j].pos, myBody.pos);
                float distSqr = r.x*r.x + r.y*r.y + r.z*r.z + SOFTENING;
                float invDist = rsqrtf(distSqr);
                float invDist3 = invDist * invDist * invDist;
                float factor = sharedBodies[j].mass * invDist3;
                acc = addFloat3(acc, scaleFloat3(r, factor));
            }
            // Process bodies after our own index.
            for (int j = threadIdx.x + 1; j < blockDim.x; j++) {
                int globalIndex = tile * blockDim.x + j;
                if (globalIndex < n) {
                    float3 r = subFloat3(sharedBodies[j].pos, myBody.pos);
                    float distSqr = r.x*r.x + r.y*r.y + r.z*r.z + SOFTENING;
                    float invDist = rsqrtf(distSqr);
                    float invDist3 = invDist * invDist * invDist;
                    float factor = sharedBodies[j].mass * invDist3;
                    acc = addFloat3(acc, scaleFloat3(r, factor));
                }
            }
        } else {
            // For tiles that do not contain our own body, no need to check self-interaction.
            #pragma unroll
            for (int j = 0; j < blockDim.x; j++) {
                int globalIndex = tile * blockDim.x + j;
                if (globalIndex < n) {
                    float3 r = subFloat3(sharedBodies[j].pos, myBody.pos);
                    float distSqr = r.x*r.x + r.y*r.y + r.z*r.z + SOFTENING;
                    float invDist = rsqrtf(distSqr);
                    float invDist3 = invDist * invDist * invDist;
                    float factor = sharedBodies[j].mass * invDist3;
                    acc = addFloat3(acc, scaleFloat3(r, factor));
                }
            }
        }
        __syncthreads();
    }

    // Scale the accumulated acceleration by the gravitational constant.
    acc = scaleFloat3(acc, G);
    // Update velocity and position with a simple Euler integration.
    myBody.vel = addFloat3(myBody.vel, scaleFloat3(acc, dt));
    myBody.pos = addFloat3(myBody.pos, scaleFloat3(myBody.vel, dt));
    new_bodies[i] = myBody;
}

void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int nBodies = 1 << 14;
    const size_t bytes = nBodies * sizeof(Body);
    const float dt = 0.01f;

    Body *h_bodies = (Body *)malloc(bytes);
    Body *h_new_bodies = (Body *)malloc(bytes);

    // Initialize bodies with random positions, velocities, and masses.
    for (int i = 0; i < nBodies; i++) {
        h_bodies[i].pos = make_float3(
            (float)rand() / RAND_MAX,
            (float)rand() / RAND_MAX,
            (float)rand() / RAND_MAX);
        h_bodies[i].vel = make_float3(
            ((float)rand() / RAND_MAX) * 0.1f,
            ((float)rand() / RAND_MAX) * 0.1f,
            ((float)rand() / RAND_MAX) * 0.1f);
        h_bodies[i].mass = ((float)rand() / RAND_MAX) + 0.1f;
    }

    Body *d_bodies, *d_new_bodies;
    checkCuda(cudaMalloc(&d_bodies, bytes), "Allocating d_bodies");
    checkCuda(cudaMalloc(&d_new_bodies, bytes), "Allocating d_new_bodies");
    checkCuda(cudaMemcpy(d_bodies, h_bodies, bytes, cudaMemcpyHostToDevice), "Copying bodies to device");

    int gridSize = (nBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t sharedMemSize = BLOCK_SIZE * sizeof(Body);

    nbodyKernel<<<gridSize, BLOCK_SIZE, sharedMemSize>>>(d_bodies, d_new_bodies, nBodies, dt);
    checkCuda(cudaGetLastError(), "Kernel launch");
    checkCuda(cudaDeviceSynchronize(), "Kernel execution");
    checkCuda(cudaMemcpy(h_new_bodies, d_new_bodies, bytes, cudaMemcpyDeviceToHost), "Copying new bodies to host");

    // Print out positions for a few bodies as a simple check.
    for (int i = 0; i < 5; i++) {
        printf("Body %d: pos = (%f, %f, %f)\n", i,
               h_new_bodies[i].pos.x,
               h_new_bodies[i].pos.y,
               h_new_bodies[i].pos.z);
    }

    free(h_bodies);
    free(h_new_bodies);
    cudaFree(d_bodies);
    cudaFree(d_new_bodies);

    return 0;
}

