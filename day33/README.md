# CUDA Q-Learning Implementation for a Grid World Environment

## 1. Introduction

This project presents a robust and modular implementation of a Q-learning algorithm accelerated using NVIDIA's CUDA framework. The primary goal is to solve a Grid World environment problem using reinforcement learning (RL) techniques while leveraging GPU parallelism to improve computational performance. Key algorithmic enhancements include the incorporation of Experience Replay and Target Networks to stabilize learning, as well as comprehensive logging and automatic plotting of results.

## 2. Problem Statement

The objective is to design an RL agent that learns an optimal policy for navigating a configurable grid environment with various cell types (e.g., obstacles, slow cells, and goal states). The agent must learn to maximize its cumulative reward while avoiding obstacles and boundaries. To achieve stable learning, the implementation integrates experience replay and periodic target network updates. Additionally, the project focuses on creating a maintainable codebase with modular components and enhanced logging for experimentation and result analysis.

## 3. Methodology

### 3.1. Environment Design

The Grid World environment is parameterized to allow flexibility in grid dimensions and cell type definitions. An `EnvironmentConfig` structure encapsulates all necessary configurations such as:
- Grid dimensions (rows and columns)
- Start and goal positions
- Reward structure for various cell types (e.g., empty, obstacle, slow, goal, boundary, step)
- Probabilities for placing obstacles and slow cells

The `GridWorld` class uses these parameters to generate a grid and copy it to device memory, ensuring that the CUDA kernels can efficiently access the environment during training.

### 3.2. Agent Design

The `QLearningAgent` class encapsulates the core RL functionality:
- **Q-Table Representation:** Two Q-tables are maintained—one for the online network and one for the target network. The target network is periodically synchronized with the online network to provide more stable target values during Q-value updates.
- **Action Selection:** An epsilon-greedy policy is employed, balancing exploration and exploitation.
- **Experience Replay:** A replay buffer stores transitions (state, action, reward, next state, done flag). Mini-batches are randomly sampled from this buffer to perform updates, thus breaking correlations between sequential transitions.
- **CUDA Kernels for Updates:** A custom CUDA kernel is defined to perform batched Q-value updates, leveraging GPU parallelism.

### 3.3. Training and Logging

The training loop executes for a configurable number of episodes. In each episode:
- The agent interacts with the environment until it reaches the goal or exceeds a maximum number of steps.
- Transitions are stored in the replay buffer.
- The agent periodically trains on mini-batches sampled from the buffer.
- The target network is updated at specified intervals.
- Epsilon is decayed over episodes to shift from exploration to exploitation.

Results are logged into a CSV file with detailed metrics such as episode number, steps taken, total reward, final state, and epsilon value. A Python script is automatically generated and executed to plot:
- Episode Reward Curve
- Steps per Episode Curve
- Epsilon Decay Curve

## 4. Implementation

The project is implemented in C++ using CUDA for GPU acceleration. The code is structured into two primary classes (`GridWorld` and `QLearningAgent`), which encapsulate environment handling and agent behavior, respectively. Key implementation highlights include:
- **Modular Code Structure:** Clear separation of environment and agent logic improves readability and maintainability.
- **Algorithmic Enhancements:** Experience replay and target network update mechanisms are implemented to enhance learning stability.
- **Logging and Visualization:** Comprehensive CSV logging and automatic plot generation facilitate experiment tracking and performance analysis.

## 5. Results and Discussion

During training, the agent progressively learns to navigate the grid while avoiding obstacles and boundary conditions. The episode reward curve typically shows an upward trend, indicating an increase in the agent's ability to maximize cumulative rewards. Similarly, the number of steps required per episode decreases as the agent converges towards the optimal policy. The epsilon decay curve confirms a gradual shift from exploratory to exploitative behavior.

The integration of experience replay and target networks proved essential in reducing the variance of updates, leading to more stable and efficient learning. Moreover, GPU acceleration via CUDA allows the system to handle larger state spaces and more complex environments efficiently.

## 6. Conclusion and Future Work

This project demonstrates a mature and flexible implementation of a CUDA-accelerated Q-learning algorithm in a grid world setting. The modular design and algorithmic enhancements contribute to improved learning stability and performance. 

Future work may include:
- Extending the environment to more complex or dynamic scenarios.
- Experimenting with advanced RL algorithms (e.g., Deep Q-Networks, Double Q-Learning).
- Further optimizing CUDA kernels for higher throughput and reduced latency.
- Integrating additional logging metrics and real-time visualization during training.

Overall, this project serves as a robust foundation for exploring reinforcement learning in GPU-accelerated environments and can be readily extended to more challenging RL problems.


