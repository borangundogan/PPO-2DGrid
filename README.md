🧠 PPO: Proximal Policy Optimization in MiniGrid Environments

This repository implements Proximal Policy Optimization (PPO) from scratch using PyTorch and tests it on lightweight MiniGrid environments. The project is part of the MERLIN (Meta- and Reinforcement Learning Infrastructure for Evaluation) initiative, aiming to build reliable and extensible RL baselines for further experimentation with Meta-RL algorithms such as MAML, FOMAML, Reptile, and LAMRL.

🚀 Features

Clean and modular PPO implementation in PyTorch

Configurable actor–critic architecture with ReLU / SiLU activations and dropout

Experiment logging and model checkpointing

Evaluation scripts for trained policies (test.py)

Compatible with gymnasium and minigrid tasks (e.g., MiniGrid-Empty-8x8-v0)

Designed for easy extension toward Meta-RL and task-based adaptation

🧩 Project Structure
src/
 ├── actor_critic.py     # Policy and value network definitions
 ├── ppo.py              # PPO algorithm core (update, clipping, advantages)
 ├── utils.py            # Helpers for rollout, normalization, etc.
train.py                 # Main training script with logging/checkpoints
test.py                  # Evaluation of trained PPO models

🧪 Example Usage
uv run python train.py --env MiniGrid-Empty-8x8-v0 --epochs 10
uv run python test.py --model_path checkpoints/MiniGrid-Empty-8x8-v0_relu_dropout0p1/ppo_model.pth

🎯 Next Steps

Integrate meta-learning algorithms (MAML, Reptile, LAMRL)

Benchmark adaptation performance across multiple grid environments

Extend logging and visualization for learning curves and success rates

📚 References

Schulman et al., Proximal Policy Optimization Algorithms, 2017

MiniGrid: Chevalier-Boisvert et al., Gym-Minigrid: A Minimalistic Gridworld Environment
