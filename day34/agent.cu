#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <time.h>
#define GRID_SIZE 32
#define NUM_CELLS (GRID_SIZE * GRID_SIZE)
#define NUM_RESOURCE_TYPES 2
#define NUM_AGENTS 1024
#define NUM_ITERATIONS 1000
#define STARVATION_LIMIT 50

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

struct __align__(16) ResourceCell {
    float resource[NUM_RESOURCE_TYPES];
    float maxResource[NUM_RESOURCE_TYPES];
    float baseRegen[NUM_RESOURCE_TYPES];
};

struct __align__(16) Agent {
    int x;
    int y;
    int type;
    float desired[NUM_RESOURCE_TYPES];
    float totalReward;
    int alive;
    int starvationCounter;
};

__device__ float atomicExtract(float *address, float desiredVal) {
    int* address_as_int = (int*)address;
    int old_int = *address_as_int;
    float old = __int_as_float(old_int);
    float extracted;
    float new_val;
    do {
        if(old < desiredVal) {
            extracted = old;
            new_val = 0.0f;
        } else {
            extracted = desiredVal;
            new_val = old - desiredVal;
        }
        int new_int = __float_as_int(new_val);
        int prev_int = atomicCAS(address_as_int, old_int, new_int);
        if(prev_int == old_int) break;
        old_int = prev_int;
        old = __int_as_float(old_int);
    } while(true);
    return extracted;
}

__global__ void initAgentKernel(Agent *agents, curandState *agentStates, unsigned int seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= NUM_AGENTS) return;
    curand_init(seed, id, 0, &agentStates[id]);
    curandState localState = agentStates[id];
    agents[id].x = curand(&localState) % GRID_SIZE;
    agents[id].y = curand(&localState) % GRID_SIZE;
    agents[id].type = curand(&localState) % 2;
    for(int i = 0; i < NUM_RESOURCE_TYPES; i++){
        agents[id].desired[i] = 1.0f + curand_uniform(&localState);
    }
    agents[id].totalReward = 0.0f;
    agents[id].alive = 1;
    agents[id].starvationCounter = 0;
    agentStates[id] = localState;
}

__global__ void initResourceKernel(ResourceCell *cells, curandState *cellStates, unsigned int seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= NUM_CELLS) return;
    curand_init(seed, id, 0, &cellStates[id]);
    curandState localState = cellStates[id];
    for(int i = 0; i < NUM_RESOURCE_TYPES; i++){
        cells[id].maxResource[i] = 10.0f + 10.0f * curand_uniform(&localState);
        cells[id].resource[i] = cells[id].maxResource[i] * curand_uniform(&localState);
        cells[id].baseRegen[i] = 0.1f + 0.2f * curand_uniform(&localState);
    }
    cellStates[id] = localState;
}

__global__ void agentKernel(Agent *agents, ResourceCell *cells, curandState *agentStates) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= NUM_AGENTS) return;
    curandState state = agentStates[id];
    if(agents[id].alive) {
        int cellIdx = agents[id].x + agents[id].y * GRID_SIZE;
        float harvested[NUM_RESOURCE_TYPES];
        int insufficient = 1;
        for(int i = 0; i < NUM_RESOURCE_TYPES; i++){
            harvested[i] = atomicExtract(&cells[cellIdx].resource[i], agents[id].desired[i]);
            agents[id].totalReward += harvested[i];
            if(harvested[i] >= 0.5f * agents[id].desired[i]) {
                insufficient = 0;
            }
        }
        if(insufficient)
            agents[id].starvationCounter++;
        else if(agents[id].starvationCounter > 0)
            agents[id].starvationCounter--;
        if(agents[id].starvationCounter > STARVATION_LIMIT)
            agents[id].alive = 0;
        float rdx = curand_uniform(&state);
        float rdy = curand_uniform(&state);
        int dx = (rdx < (1.0f/3.0f)) ? -1 : (rdx < (2.0f/3.0f) ? 0 : 1);
        int dy = (rdy < (1.0f/3.0f)) ? -1 : (rdy < (2.0f/3.0f) ? 0 : 1);
        int newX = (agents[id].x + dx + GRID_SIZE) % GRID_SIZE;
        int newY = (agents[id].y + dy + GRID_SIZE) % GRID_SIZE;
        agents[id].x = newX;
        agents[id].y = newY;
        int newCell = newX + newY * GRID_SIZE;
        for(int i = 0; i < NUM_RESOURCE_TYPES; i++){
            float clampVal = 0.5f * cells[newCell].maxResource[i];
            if(agents[id].desired[i] > clampVal)
                agents[id].desired[i] = clampVal;
        }
    }
    agentStates[id] = state;
}

__global__ void regenKernel(ResourceCell *cells, curandState *cellStates) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= NUM_CELLS) return;
    curandState state = cellStates[id];
    for(int i = 0; i < NUM_RESOURCE_TYPES; i++){
        float current = cells[id].resource[i];
        float maxVal = cells[id].maxResource[i];
        float regenFactor = cells[id].baseRegen[i] * (1.0f - current / maxVal);
        float noise = 0.05f * (curand_uniform(&state) - 0.5f);
        float regen = regenFactor + noise;
        float depletion = 0.01f * current;
        float newVal = current + regen - depletion;
        if(newVal < 0.0f) newVal = 0.0f;
        if(newVal > maxVal) newVal = maxVal;
        cells[id].resource[i] = newVal;
    }
    cellStates[id] = state;
}

int main(){
    Agent *d_agents;
    ResourceCell *d_cells;
    curandState *d_agentStates, *d_cellStates;
    CHECK_CUDA(cudaMalloc(&d_agents, NUM_AGENTS * sizeof(Agent)));
    CHECK_CUDA(cudaMalloc(&d_cells, NUM_CELLS * sizeof(ResourceCell)));
    CHECK_CUDA(cudaMalloc(&d_agentStates, NUM_AGENTS * sizeof(curandState)));
    CHECK_CUDA(cudaMalloc(&d_cellStates, NUM_CELLS * sizeof(curandState)));
    
    int blockSize = 256;
    int numAgentBlocks = (NUM_AGENTS + blockSize - 1) / blockSize;
    int numCellBlocks = (NUM_CELLS + blockSize - 1) / blockSize;
    unsigned int seed = (unsigned int) time(NULL);
    
    initAgentKernel<<<numAgentBlocks, blockSize>>>(d_agents, d_agentStates, seed);
    CHECK_CUDA(cudaGetLastError());
    initResourceKernel<<<numCellBlocks, blockSize>>>(d_cells, d_cellStates, seed);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    for(int iter = 0; iter < NUM_ITERATIONS; iter++){
        agentKernel<<<numAgentBlocks, blockSize>>>(d_agents, d_cells, d_agentStates);
        CHECK_CUDA(cudaGetLastError());
        regenKernel<<<numCellBlocks, blockSize>>>(d_cells, d_cellStates);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        
        if(iter % 100 == 0 || iter == NUM_ITERATIONS - 1) {
            Agent *h_agentsTemp = (Agent*)malloc(NUM_AGENTS * sizeof(Agent));
            ResourceCell *h_cellsTemp = (ResourceCell*)malloc(NUM_CELLS * sizeof(ResourceCell));
            CHECK_CUDA(cudaMemcpy(h_agentsTemp, d_agents, NUM_AGENTS * sizeof(Agent), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_cellsTemp, d_cells, NUM_CELLS * sizeof(ResourceCell), cudaMemcpyDeviceToHost));
            
            int aliveCount = 0;
            float totalReward = 0.0f;
            for(int i = 0; i < NUM_AGENTS; i++){
                if(h_agentsTemp[i].alive){
                    aliveCount++;
                    totalReward += h_agentsTemp[i].totalReward;
                }
            }
            float avgReward = (aliveCount > 0) ? totalReward / aliveCount : 0.0f;
            float avgResource[NUM_RESOURCE_TYPES] = {0.0f};
            for (int i = 0; i < NUM_CELLS; i++){
                for (int j = 0; j < NUM_RESOURCE_TYPES; j++){
                    avgResource[j] += h_cellsTemp[i].resource[j];
                }
            }
            for(int j = 0; j < NUM_RESOURCE_TYPES; j++){
                avgResource[j] /= NUM_CELLS;
            }
            printf("Iteration %d: Alive Agents = %d, Avg Reward = %f, Avg Resources = ", iter, aliveCount, avgReward);
            for(int j = 0; j < NUM_RESOURCE_TYPES; j++){
                printf("%f ", avgResource[j]);
            }
            printf("\n");
            free(h_agentsTemp);
            free(h_cellsTemp);
        }
    }
    
    Agent *h_agents = (Agent*)malloc(NUM_AGENTS * sizeof(Agent));
    ResourceCell *h_cells = (ResourceCell*)malloc(NUM_CELLS * sizeof(ResourceCell));
    CHECK_CUDA(cudaMemcpy(h_agents, d_agents, NUM_AGENTS * sizeof(Agent), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_cells, d_cells, NUM_CELLS * sizeof(ResourceCell), cudaMemcpyDeviceToHost));
    
    float totalReward = 0.0f;
    int aliveCount = 0;
    for(int i = 0; i < NUM_AGENTS; i++){
        totalReward += h_agents[i].totalReward;
        if(h_agents[i].alive)
            aliveCount++;
    }
    printf("Final: Total Reward = %f, Alive Agents = %d\n", totalReward, aliveCount);
    
    free(h_agents);
    free(h_cells);
    CHECK_CUDA(cudaFree(d_agents));
    CHECK_CUDA(cudaFree(d_cells));
    CHECK_CUDA(cudaFree(d_agentStates));
    CHECK_CUDA(cudaFree(d_cellStates));
    return 0;
}

