# src/analyze_tasks.py

from __future__ import annotations
import argparse
import os
from typing import Dict
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as ticker

from src.scenario_creator.scenario_creator import ScenarioCreator
from src.metrics.task_metrics import compare_two_feature_sets
from src.utils.utils import get_device, set_seed
from src.actor_critic import MLPActorCritic, CNNActorCritic


# Collect deterministic reward distributions
def collect_rewards(env, policy, device, num_episodes=50, seed=0):
    rewards = []
    base = seed

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=base + ep)
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
                action_tensor, _, _ = policy.act(obs_t, deterministic=True)
                action = action_tensor.item()

            obs, reward, terminated, truncated, _ = env.step(action)
            obs = np.array(obs, dtype=np.float32)

            ep_reward += reward
            done = terminated or truncated

        rewards.append(ep_reward)

    return np.array(rewards)


# Load PPO model
def load_policy(model_path, sample_obs, act_dim, device):
    if sample_obs.ndim == 1:
        obs_dim = int(np.prod(sample_obs.shape))
        policy = MLPActorCritic(obs_dim, act_dim).to(device)
    else:
        policy = CNNActorCritic(sample_obs.shape, act_dim).to(device)

    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()
    return policy

def plot_reward_distribution(r1, r2, name1, name2, save_path):
    plt.figure(figsize=(10, 6))
    
    data_df = pd.concat([
        pd.DataFrame({"Episode Return": r1, "Task": name1}),
        pd.DataFrame({"Episode Return": r2, "Task": name2})
    ])
    
    sns.histplot(
        data=data_df,
        x="Episode Return",
        hue="Task",
        stat="probability",  
        kde=False,            
        bins=30,        
        binrange=(0, 1),     
        multiple="layer",    
        element="step",      
        palette=["#1f77b4", "#ff7f0e"], 
        alpha=0.5,          
        common_norm=False,    
        linewidth=0.5         
    )
    
    plt.ylabel("Probability", fontsize=12)
    plt.xlabel("Episode Return", fontsize=12)
    plt.title(f"Reward Distribution: {name1} vs {name2}", fontsize=14)
    
    plt.ylim(-0.02, 1.05)
    plt.grid(True, linestyle=':', alpha=0.6) 
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# Bar chart of mean episode rewards
def plot_reward_bars(reward_dict, save_path):
    tasks = list(reward_dict.keys())
    means = [np.mean(reward_dict[t]) for t in tasks]
    stds = [np.std(reward_dict[t]) for t in tasks]

    plt.figure(figsize=(7, 5))
    plt.bar(tasks, means, yerr=stds, capsize=6)
    plt.title("Mean Episode Reward per Task")
    plt.ylabel("Mean Reward")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--difficulties", nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=999, help="Evaluation seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device("auto")

    # Model path structure: checkpoints / Experiment_Name / Seed_X / ppo_model.pth
    path_parts = args.model_path.strip("/").split("/")
    
    if len(path_parts) >= 3:
        experiment_name = path_parts[-3]
        seed_name = path_parts[-2]
        out_dir = os.path.join("analysis_results", experiment_name, seed_name)
    else:
        # Fallback for simpler paths
        model_name = path_parts[-2]
        out_dir = f"analysis_results/{model_name}"

    os.makedirs(out_dir, exist_ok=True)
    print(f"[Output] Saving results to: {out_dir}")

    # Scenario creator
    sc = ScenarioCreator("src/config/scenario.yaml")

    reward_dict: Dict[str, np.ndarray] = {}

    # Collect reward sequences for each task
    for diff in args.difficulties:
        print(f"[Collecting] {diff}")

        env = sc.create_env(diff, seed=args.seed)

        sample_obs, _ = env.reset(seed=args.seed)
        sample_obs = np.array(sample_obs, dtype=np.float32)
        act_dim = env.action_space.n

        policy = load_policy(args.model_path, sample_obs, act_dim, device)

        rewards = collect_rewards(
            env,
            policy,
            device,
            num_episodes=args.episodes,
            seed=args.seed
        )

        reward_dict[diff] = rewards

    # Save bar chart
    plot_reward_bars(reward_dict, f"{out_dir}/reward_bar_chart.png")

    # Pairwise comparisons
    keys = list(reward_dict.keys())
    print("\n===== TASK DISTRIBUTION METRICS (REWARD-BASED) =====\n")

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            r1, r2 = reward_dict[a], reward_dict[b]

            # Generate probability distribution plot (replaces old KDE)
            plot_reward_distribution(r1, r2, a, b, f"{out_dir}/{a}_vs_{b}_reward_dist.png")

            # Compute distance metrics
            metrics = compare_two_feature_sets(
                r1.reshape(-1, 1),
                r2.reshape(-1, 1)
            )

            print(f"{a} vs {b}")
            for k, v in metrics.items():
                print(f"   {k:20s}: {v:.6f}")
            print()

    print(f"\nSaved all outputs to: {out_dir}/")

if __name__ == "__main__":
    main()