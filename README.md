# PPO: Proximal Policy Optimization in MiniGrid Environments

This repository implements **Proximal Policy Optimization (PPO)** from scratch using PyTorch and tests it on lightweight **MiniGrid** environments. The project is part of the *MERLIN* (Meta- and Reinforcement Learning Infrastructure for Evaluation) initiative, aiming to build reliable and extensible RL baselines for further experimentation with **Meta-RL algorithms** such as MAML, FOMAML, Reptile, and LAMRL.

## Features
- Clean and modular PPO implementation in PyTorch
- Configurable actor–critic architecture with ReLU / SiLU activations and dropout
- Experiment logging and model checkpointing
- Evaluation scripts for trained policies (`test.py`)
- Compatible with `gymnasium` and `minigrid` tasks (e.g., `MiniGrid-Empty-8x8-v0`)
- Designed for easy extension toward **Meta-RL and task-based adaptation**

## Project Structure
src/
- actor_critic.py # Policy and value network definitions
- ppo.py # PPO algorithm core (update, clipping, advantages)
- utils.py # Helpers for rollout, normalization, etc.
- train.py # Main training script with logging/checkpoints
- test.py # Evaluation of trained PPO models

## Example Usage
```bash
uv run python train.py --env MiniGrid-Empty-8x8-v0 --epochs 10
uv run python test.py --model_path checkpoints/MiniGrid-Empty-8x8-v0_relu_dropout0p1/ppo_model.pth
```

## Next Steps
- Integrate meta-learning algorithms (MAML, Reptile, LAMRL)
- Benchmark adaptation performance across multiple grid environments
- Extend logging and visualization for learning curves and success rates

## References
- Schulman et al., Proximal Policy Optimization Algorithms, 2017
- Chevalier-Boisvert et al., Gym-Minigrid: A Minimalistic Gridworld Environment
- Y. Li, Deep Reinforcement Learning: An Overview, 2018.
- Chelsea Finn, Pieter Abbeel und Sergey Levine, "Model-Agnostic Meta-Learning for
Fast Adaptation of Deep Networks," International Conference on Machine Learning, S.
1126–1135, 2017. [Online]. Verfügbar unter: https://proceedings.mlr.press/v70/
finn17a.html
- X. Tian, H. Zhou, H. Zhang, L. Zhang und L. Shi, "LSTM & attention-based meta-
reinforcement learning for trajectory tracking of underwater gliders with varying
buoyancy loss and current disturbance," Ocean Engineering, Jg. 326, S. 120906, 2025.
doi: 10.1016/j.oceaneng.2025.120906. [Online]. Verfügbar unter: https://
www.sciencedirect.com/science/article/pii/S0029801825006195
- Minigrid & miniworld: Modular & customizable reinforcement learning environments
for goal-oriented tasks, 2023.
- M. A. Ali, A. Maqsood, U. Athar und H. R. Khanzada, "Comparative Evaluation of
Reinforcement Learning Algorithms for Multi-Agent Unmanned Aerial Vehicle Path
Planning in 2D and 3D Environments," Drones, Jg. 9, Nr. 6, S. 438, 2025. doi:
10.3390/drones9060438. [Online]. Verfügbar unter: https://www.mdpi.com/2504-446X/
9/6/438

