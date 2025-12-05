# src/analyze_tasks.py

from __future__ import annotations
import argparse
import os
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from src.scenario_creator.scenario_creator import ScenarioCreator
from src.metrics.task_metrics import compare_two_feature_sets
from src.utils import get_device
from src.actor_critic import MLPActorCritic, CNNActorCritic


# Reward toplama (artık feature extraction yok)
def collect_rewards(env, policy, device, num_episodes=50):
    rewards = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        obs = np.array(obs, dtype=np.float32)
        done = False
        ep_reward = 0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

            if obs.ndim == 3:
                obs_t = obs_t.unsqueeze(0) / 255.0
            else:
                obs_t = obs_t.view(1, -1) / 255.0

            with torch.no_grad():
                logits = policy.actor(obs_t)
                action = torch.argmax(logits).item()

            obs, reward, terminated, truncated, _ = env.step(action)
            obs = np.array(obs, dtype=np.float32)

            ep_reward += reward
            done = terminated or truncated

        rewards.append(ep_reward)

    return np.array(rewards)



# PPO model
def load_policy(model_path, sample_obs, act_dim, device):
    if sample_obs.ndim == 1:
        obs_dim = int(np.prod(sample_obs.shape))
        policy = MLPActorCritic(obs_dim, act_dim).to(device)
    else:
        policy = CNNActorCritic(sample_obs.shape, act_dim).to(device)

    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()
    return policy


# Reward-based KDE plot
def plot_reward_kde(r1, r2, name1, name2, save_path):
    plt.figure(figsize=(7,5))
    sns.kdeplot(r1, label=name1, linewidth=2)
    sns.kdeplot(r2, label=name2, linewidth=2)
    plt.title(f"Reward Distribution: {name1} vs {name2}")
    plt.xlabel("Episode Return")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# Bar chart of mean rewards
def plot_reward_bars(reward_dict, save_path):
    tasks = list(reward_dict.keys())
    means = [np.mean(reward_dict[t]) for t in tasks]
    stds  = [np.std(reward_dict[t]) for t in tasks]

    plt.figure(figsize=(7,5))
    plt.bar(tasks, means, yerr=stds, capsize=6)
    plt.title("Mean Episode Reward per Task")
    plt.ylabel("Mean Reward")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--difficulties", nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    device = get_device("auto")

    # Output directory
    model_name = args.model_path.split("/")[-2]
    out_dir = f"analysis_results/{model_name}"
    os.makedirs(out_dir, exist_ok=True)

    # Scenario creator
    sc = ScenarioCreator("src/config/scenario.yaml")

    reward_dict: Dict[str, np.ndarray] = {}

    # Collect reward distributions
    for diff in args.difficulties:
        print(f"[Collecting] {diff}")

        env = sc.create_env(diff)
        sample_obs, _ = env.reset()
        sample_obs = np.array(sample_obs, dtype=np.float32)
        act_dim = env.action_space.n

        policy = load_policy(args.model_path, sample_obs, act_dim, device)

        rewards = collect_rewards(env, policy, device, num_episodes=args.episodes)
        reward_dict[diff] = rewards

    # Save bar chart
    plot_reward_bars(reward_dict, f"{out_dir}/reward_bar_chart.png")

    # Pairwise comparison + KDE + metrics
    keys = list(reward_dict.keys())
    print("\n===== TASK DISTRIBUTION METRICS (REWARD-BASED) =====\n")

    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            a, b = keys[i], keys[j]
            r1, r2 = reward_dict[a], reward_dict[b]

            plot_reward_kde(r1, r2, a, b, f"{out_dir}/{a}_vs_{b}_reward_kde.png")

            metrics = compare_two_feature_sets(
                r1.reshape(-1,1),   # reward is 1D → reshape to (N,1)
                r2.reshape(-1,1)
            )

            print(f"{a} vs {b}")
            for k, v in metrics.items():
                print(f"   {k:20s}: {v:.6f}")
            print()

    print(f"\nSaved all figures to: {out_dir}/")


if __name__ == "__main__":
    main()
