# Q-Learning Gridworld Agent

This repository contains an implementation of a Q-learning agent that learns to navigate a grid-based environment. The agent is trained to reach a goal state while avoiding obstacles using reinforcement learning techniques. Various learning scenarios, including outcome devaluation, contingency degradation, and novel goal adaptation, are tested.

## Table of Contents
- [Overview](#overview)
- [Environment](#environment)
- [Q-Learning Parameters](#q-learning-parameters)
- [Training and Testing](#training-and-testing)
- [Evaluation](#evaluation)
- [Scenarios Tested](#scenarios-tested)
- [Dependencies](#dependencies)
- [Usage](#usage)

## Overview
This project uses the Q-learning algorithm to train an agent in a gridworld environment. The agent explores the grid, learning an optimal policy to reach the goal while minimizing penalties. The project visualizes the learned policy and evaluates the agent's performance in different scenarios.

## Environment
The environment is a grid-based world with:
- **Grid Size:** `6x6`
- **Start State:** `(0, 0)`
- **Goal State:** `(5, 5)`
- **Obstacles:** Empty by default, can be modified.
- **Rewards:** Positive reward for reaching the goal, and negative step cost.

### Reward Structure
- **Goal Reward:** `25`
- **Step Cost:** `-1`

## Q-Learning Parameters
- **Alpha (Learning Rate):** `0.256` (default), varies in experiments.
- **Gamma (Discount Factor):** `0.9` (future reward consideration).
- **Epsilon (Exploration Rate):** `0.25` (for epsilon-greedy exploration).
- **Number of Episodes:** `10,000` (training episodes).
- **Max Steps per Episode:** `10,000` (limits infinite loops).

## Training and Testing
### Training
The agent is trained using Q-learning:
1. **Initialize** Q-table with zeros.
2. **Choose an action** using an epsilon-greedy strategy.
3. **Update the Q-value** using the Bellman equation.
4. **Repeat** until convergence or episode limit.

### Testing
Once trained, the agent follows the learned policy to navigate from the start state to the goal.

## Evaluation
Performance metrics include:
- **Success Rate** (percentage of successful runs reaching the goal).
- **Average Steps to Goal** (efficiency of navigation).
- **Average Reward per Episode** (effectiveness of learned policy).

## Scenarios Tested
1. **Standard Training:** The agent learns to reach the goal.
2. **Outcome Devaluation:** The goal reward is removed or set to negative.
3. **Contingency Degradation:** The goal state is removed.
4. **Novel Goal:** The goal is moved to a new location.
5. **Obstacle Introduced:** A new obstacle is placed near the goal.
6. **Exploration of Different Alpha Values:** Testing different learning rates.
7. **Random Agent:** A baseline performance comparison with a random policy.

For the following scenarios, the agent goes under a short re-training (starting from the previous policy). The training inclueds:
- **Number of Episodes:** `100` (training episodes).
- **Max Steps per Episode:** `1,000` (limits infinite loops).

## Dependencies
Ensure you have the following Python libraries installed:
```bash
pip install numpy matplotlib seaborn
```

## Usage
Run the script to train the agent and visualize the results:
```bash
python grid_world.py
```
The script will output the learned policy, agent paths, and evaluation metrics.

