#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cmath>
#include <ctime>
#include <vector>
#include <fstream>
#include <iostream>
#include <random>
#include <chrono>
#include <string>
#include <algorithm>

struct EnvironmentConfig {
    int rows;
    int cols;
    int startState;
    int goalState;
    float rewardEmpty;
    float rewardObstacle;
    float rewardGoal;
    float rewardSlow;
    float rewardBoundary;
    float rewardStep;
    float obstacleProb;
    float slowProb;
};

class GridWorld {
public:
    int rows, cols, numStates;
    std::vector<int> grid;
    EnvironmentConfig config;
    int startState, goalState;
    int *d_grid;

    GridWorld(const EnvironmentConfig& envConfig) : config(envConfig) {
        rows = config.rows;
        cols = config.cols;
        numStates = rows * cols;
        grid.resize(numStates, 0);
        startState = config.startState;
        goalState = config.goalState;
        grid[startState] = 3;
        grid[goalState] = 2;
        std::mt19937 rng((unsigned int)std::chrono::system_clock::now().time_since_epoch().count());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (int i = 0; i < numStates; i++) {
            if(i == startState || i == goalState) continue;
            float prob = dist(rng);
            if(prob < config.obstacleProb) grid[i] = 1;
            else if(prob < config.obstacleProb + config.slowProb) grid[i] = 4;
            else grid[i] = 0;
        }
        cudaMalloc(&d_grid, numStates * sizeof(int));
        cudaMemcpy(d_grid, grid.data(), numStates * sizeof(int), cudaMemcpyHostToDevice);
    }

    ~GridWorld() {
        cudaFree(d_grid);
    }

    int step(int currentState, int action) {
        int row = currentState / cols;
        int col = currentState % cols;
        int new_row = row, new_col = col;
        if(action == 0) new_row = row - 1;
        else if(action == 1) new_row = row + 1;
        else if(action == 2) new_col = col - 1;
        else if(action == 3) new_col = col + 1;
        else if(action == 4) { new_row = row - 1; new_col = col - 1; }
        else if(action == 5) { new_row = row - 1; new_col = col + 1; }
        else if(action == 6) { new_row = row + 1; new_col = col - 1; }
        else if(action == 7) { new_row = row + 1; new_col = col + 1; }
        if(new_row < 0 || new_row >= rows || new_col < 0 || new_col >= cols)
            return currentState;
        int nextState = new_row * cols + new_col;
        return nextState;
    }

    float getReward(int currentState, int nextState) {
        if(currentState == nextState) {
            int cellType = grid[nextState];
            if(cellType == 1) return config.rewardObstacle;
            else return config.rewardBoundary;
        }
        int cellType = grid[nextState];
        if(cellType == 0) return config.rewardEmpty;
        else if(cellType == 1) return config.rewardObstacle;
        else if(cellType == 2) return config.rewardGoal;
        else if(cellType == 4) return config.rewardSlow;
        else return config.rewardStep;
    }
};

struct Transition {
    int state;
    int action;
    float reward;
    int nextState;
    bool done;
};

__global__ void updateQKernelBatch(const int *states, const int *actions, const float *rewards,
                                     const int *nextStates, const int *dones, float *d_onlineQ,
                                     const float *d_targetQ, float alpha, float gamma, int numActions, int batchSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < batchSize) {
        int s = states[i];
        int a = actions[i];
        float r = rewards[i];
        int s_next = nextStates[i];
        int done = dones[i];
        float maxQ = -1e20f;
        for (int j = 0; j < numActions; j++) {
            float q = d_targetQ[s_next * numActions + j];
            if(q > maxQ) { maxQ = q; }
        }
        float target = r;
        if(!done) {
            target += gamma * maxQ;
        }
        int index = s * numActions + a;
        d_onlineQ[index] = d_onlineQ[index] + alpha * (target - d_onlineQ[index]);
    }
}

class QLearningAgent {
public:
    int numStates, numActions;
    float alpha, gamma, epsilon, minEpsilon, epsilonDecay;
    int updateTargetEvery, replayBatchSize, replayCapacity;
    std::vector<Transition> replayBuffer;
    float *d_onlineQ, *d_targetQ;

    QLearningAgent(int numStates, int numActions, float alpha, float gamma, float initialEpsilon,
                   float minEpsilon, float epsilonDecay, int updateTargetEvery,
                   int replayCapacity, int replayBatchSize)
    : numStates(numStates), numActions(numActions), alpha(alpha), gamma(gamma),
      epsilon(initialEpsilon), minEpsilon(minEpsilon), epsilonDecay(epsilonDecay),
      updateTargetEvery(updateTargetEvery), replayCapacity(replayCapacity), replayBatchSize(replayBatchSize)
    {
        cudaMalloc(&d_onlineQ, numStates * numActions * sizeof(float));
        cudaMalloc(&d_targetQ, numStates * numActions * sizeof(float));
        std::vector<float> qInit(numStates * numActions);
        std::mt19937 rng((unsigned int)std::chrono::system_clock::now().time_since_epoch().count());
        std::uniform_real_distribution<float> dist(-0.01f, 0.01f);
        for(auto &val : qInit) { val = dist(rng); }
        cudaMemcpy(d_onlineQ, qInit.data(), numStates * numActions * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_targetQ, qInit.data(), numStates * numActions * sizeof(float), cudaMemcpyHostToDevice);
    }

    ~QLearningAgent() {
        cudaFree(d_onlineQ);
        cudaFree(d_targetQ);
    }

    int chooseAction(int state) {
        std::mt19937 rng((unsigned int)std::chrono::system_clock::now().time_since_epoch().count());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float randomValue = dist(rng);
        if(randomValue < epsilon) {
            std::uniform_int_distribution<int> actionDist(0, numActions - 1);
            return actionDist(rng);
        } else {
            std::vector<float> qValues(numActions);
            cudaMemcpy(qValues.data(), d_onlineQ + state * numActions, numActions * sizeof(float), cudaMemcpyDeviceToHost);
            int bestAction = 0;
            float bestValue = qValues[0];
            for (int a = 1; a < numActions; a++) {
                if(qValues[a] > bestValue) { bestValue = qValues[a]; bestAction = a; }
            }
            return bestAction;
        }
    }

    void storeTransition(const Transition &trans) {
        if(replayBuffer.size() >= (size_t)replayCapacity) {
            replayBuffer.erase(replayBuffer.begin());
        }
        replayBuffer.push_back(trans);
    }

    void trainOnBatch() {
        if(replayBuffer.size() < (size_t)replayBatchSize) return;
        std::vector<int> states(replayBatchSize);
        std::vector<int> actions(replayBatchSize);
        std::vector<float> rewards(replayBatchSize);
        std::vector<int> nextStates(replayBatchSize);
        std::vector<int> dones(replayBatchSize);
        std::mt19937 rng((unsigned int)std::chrono::system_clock::now().time_since_epoch().count());
        std::uniform_int_distribution<size_t> dist(0, replayBuffer.size() - 1);
        for(int i = 0; i < replayBatchSize; i++){
            size_t index = dist(rng);
            const Transition &trans = replayBuffer[index];
            states[i] = trans.state;
            actions[i] = trans.action;
            rewards[i] = trans.reward;
            nextStates[i] = trans.nextState;
            dones[i] = trans.done ? 1 : 0;
        }
        int *d_states, *d_actions, *d_nextStates, *d_dones;
        float *d_rewards;
        cudaMalloc(&d_states, replayBatchSize * sizeof(int));
        cudaMalloc(&d_actions, replayBatchSize * sizeof(int));
        cudaMalloc(&d_rewards, replayBatchSize * sizeof(float));
        cudaMalloc(&d_nextStates, replayBatchSize * sizeof(int));
        cudaMalloc(&d_dones, replayBatchSize * sizeof(int));
        cudaMemcpy(d_states, states.data(), replayBatchSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_actions, actions.data(), replayBatchSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rewards, rewards.data(), replayBatchSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nextStates, nextStates.data(), replayBatchSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dones, dones.data(), replayBatchSize * sizeof(int), cudaMemcpyHostToDevice);
        int threads = 256;
        int blocks = (replayBatchSize + threads - 1) / threads;
        updateQKernelBatch<<<blocks, threads>>>(d_states, d_actions, d_rewards, d_nextStates, d_dones,
                                                  d_onlineQ, d_targetQ, alpha, gamma, numActions, replayBatchSize);
        cudaDeviceSynchronize();
        cudaFree(d_states);
        cudaFree(d_actions);
        cudaFree(d_rewards);
        cudaFree(d_nextStates);
        cudaFree(d_dones);
    }

    void updateTargetNetwork() {
        cudaMemcpy(d_targetQ, d_onlineQ, numStates * numActions * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    void decayEpsilon() {
        epsilon = std::max(minEpsilon, epsilon * epsilonDecay);
    }
};

int main(){
    EnvironmentConfig envConfig;
    envConfig.rows = 10;
    envConfig.cols = 10;
    envConfig.startState = 0;
    envConfig.goalState = envConfig.rows * envConfig.cols - 1;
    envConfig.rewardEmpty = -1.0f;
    envConfig.rewardObstacle = -50.0f;
    envConfig.rewardGoal = 100.0f;
    envConfig.rewardSlow = -3.0f;
    envConfig.rewardBoundary = -10.0f;
    envConfig.rewardStep = -1.0f;
    envConfig.obstacleProb = 0.2f;
    envConfig.slowProb = 0.1f;

    GridWorld gridWorld(envConfig);
    int numStates = gridWorld.numStates;
    int numActions = 8;
    int numEpisodes = 2000;
    int maxStepsPerEpisode = 200;

    QLearningAgent agent(numStates, numActions, 0.1f, 0.9f, 0.9f, 0.1f, 0.995f, 50, 10000, 32);

    std::ofstream logFile("training_log.csv");
    logFile << "Episode,Steps,TotalReward,FinalState,Epsilon\n";

    for(int episode = 0; episode < numEpisodes; episode++){
        int currentState = gridWorld.startState;
        float totalReward = 0.0f;
        int steps = 0;
        bool done = false;
        for(steps = 0; steps < maxStepsPerEpisode; steps++){
            int action = agent.chooseAction(currentState);
            int nextState = gridWorld.step(currentState, action);
            float reward = gridWorld.getReward(currentState, nextState);
            if(nextState == gridWorld.goalState) done = true;
            Transition trans { currentState, action, reward, nextState, done };
            agent.storeTransition(trans);
            totalReward += reward;
            currentState = nextState;
            if(done) { steps++; break; }
        }
        agent.trainOnBatch();
        if(episode % agent.updateTargetEvery == 0) {
            agent.updateTargetNetwork();
        }
        agent.decayEpsilon();
        logFile << episode << "," << steps << "," << totalReward << "," << currentState << "," << agent.epsilon << "\n";
        std::cout << "Episode " << episode << ": Steps = " << steps << ", Total Reward = " << totalReward
                  << ", Final State = " << currentState << ", Epsilon = " << agent.epsilon << "\n";
    }
    logFile.close();

    std::ofstream plotScript("plot_results.py");
    plotScript << "import pandas as pd\n";
    plotScript << "import matplotlib.pyplot as plt\n";
    plotScript << "data = pd.read_csv('training_log.csv')\n";
    plotScript << "plt.figure(); plt.plot(data['Episode'], data['TotalReward']); plt.xlabel('Episode'); plt.ylabel('Total Reward'); plt.title('Episode Reward Curve'); plt.savefig('episode_reward.png')\n";
    plotScript << "plt.figure(); plt.plot(data['Episode'], data['Steps']); plt.xlabel('Episode'); plt.ylabel('Steps'); plt.title('Steps per Episode'); plt.savefig('steps_per_episode.png')\n";
    plotScript << "plt.figure(); plt.plot(data['Episode'], data['Epsilon']); plt.xlabel('Episode'); plt.ylabel('Epsilon'); plt.title('Epsilon Decay Curve'); plt.savefig('epsilon_decay.png')\n";
    plotScript.close();
    
    system("python plot_results.py");
    return 0;
}

