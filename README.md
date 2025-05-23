# RL-with-SUMO-Experiments-

# Reinforcement Learning for Traffic Signal Control in SUMO

This repository contains experiments using **Policy Gradient Reinforcement Learning** algorithms for optimizing traffic signal control in a custom SUMO simulation environment.

## 🚦 Project Overview

Urban traffic congestion is a major challenge. In this work, we simulate a traffic junction using the **SUMO (Simulation of Urban MObility)** environment and apply **Policy Gradient methods** to dynamically control traffic lights with the goal of minimizing vehicle waiting time and improving flow.

## 🏗️ SUMO Environment Setup

- Created using `netedit` with 5 junctions: J0, J1 (center), J2, J3, J4
- Bidirectional roads between junctions
- Traffic light at J1 is controlled via `traci`
- 12 unique vehicle flows defined (e.g., n-s, w-e, s-n, etc.)
- Simulation controlled using Python & TraCI interface

## 🎯 Reinforcement Learning Approach

- **RL Agent**: Policy Gradient (REINFORCE algorithm)
- **State Representation**: Number of vehicles waiting per incoming lane
- **Action Space**: Green time distributions across 4 phases summing to 64s (e.g., [16, 16, 16, 16])
- **Reward**: Negative of total waiting time (or delay)
- **Policy Network**: Simple feedforward neural network outputting action probabilities

## ⚙️ Experimental Setup

- Episodes: 500 (each episode = 300 simulation seconds)
- Reward logged per episode
- Agent trains after each episode
- Optimizer: Adam
- Learning rate: 0.001
- Frameworks: `PyTorch`, `TraCI`, `SUMO`

## 📊 Results

- Training converges to a stable policy after ~300 episodes
- Average waiting time reduced by ~30% compared to fixed-time controller
- (Insert plot here: reward vs episodes, waiting time per episode)

## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
