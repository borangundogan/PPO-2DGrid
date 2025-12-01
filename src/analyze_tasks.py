# src/analyze_tasks.py

from __future__ import annotations
import argparse
import os
from datetime import datetime
from typing import Dict

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from src.scenario_creator.scenario_creator import ScenarioCreator
from src.metrics.task_metrics import compare_task_feature_dict
from src.utils import get_device
from src.actor_critic import MLPActorCritic, CNNActorCritic


# ===============================================================
# Utility: extract logits as features
# ===============================================================
def extract_feature_vector(policy, obs_t):
    """Return actor logits (pre-softmax), which exist for both MLP and CNN."""
    with torch.no_grad():
        if isinstance(policy, MLPActorCritic):
            x = policy.actor[:-1](obs_t)    # up to last Linear
            logits = policy.actor[-1](x)    # final Linear layer
            return logits.cpu().numpy()[0]

        else:
            raise ValueError("Unknown policy class.")


# ===============================================================
# Collect features from environment
# ===============================================================
def collect_features(env, policy, device, num_steps=2000):
    obs, _ = env.reset()
    obs = np.array(obs, dtype=np.float32)

    feats = []
    for _ in range(num_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        if obs.ndim == 3:       # image
            obs_t = obs_t.unsqueeze(0) / 255.0
        else:
            obs_t = obs_t.view(1, -1) / 255.0

        feats.append(extract_feature_vector(policy, obs_t))

        # step with greedy policy
        with torch.no_grad():
            if obs.ndim == 3:
                logits = policy.actor(policy.cnn(obs_t.permute(0,3,1,2)))
            else:
                logits = policy.actor(obs_t)
            action = torch.argmax(logits).item()

        obs, _, terminated, truncated, _ = env.step(action)
        obs = np.array(obs, dtype=np.float32)
        if terminated or truncated:
            obs, _ = env.reset()
            obs = np.array(obs, dtype=np.float32)

    return np.array(feats)   # (N, act_dim)


# ===============================================================
# Load policy
# ===============================================================
def load_policy(model_path, sample_obs, act_dim, device):
    if sample_obs.ndim == 1:
        obs_dim = int(np.prod(sample_obs.shape))
        policy = MLPActorCritic(obs_dim, act_dim).to(device)
    else:
        obs_shape = sample_obs.shape
        policy = CNNActorCritic(obs_shape, act_dim).to(device)

    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()
    return policy


# ===============================================================
# Visualization utilities
# ===============================================================
def plot_kde(feats_a, feats_b, name_a, name_b, save_path):
    """Single-dimension KDE over mean of logits."""
    p = feats_a.mean(axis=1)
    q = feats_b.mean(axis=1)

    plt.figure(figsize=(7,5))
    sns.kdeplot(p, label=name_a, color="blue", linewidth=2)
    sns.kdeplot(q, label=name_b, color="orange", linewidth=2)
    plt.title(f"Distribution Comparison: {name_a} vs {name_b}")
    plt.xlabel("Mean Logit Activation")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_mean_std(feats_a, feats_b, name_a, name_b, save_path):
    """Plot smooth mean curves like second figure."""
    mean_a = feats_a.mean(axis=0)
    mean_b = feats_b.mean(axis=0)

    x = np.arange(len(mean_a))

    plt.figure(figsize=(7,5))
    plt.plot(x, mean_a, label=f"{name_a}", color="red")
    plt.plot(x, mean_b, label=f"{name_b}", color="green")
    plt.title(f"Mean Comparison: {name_a} vs {name_b}")
    plt.xlabel("Feature dimension")
    plt.ylabel("Mean activation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ===============================================================
# Main Analyze Script
# ===============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--difficulties", nargs="+", required=True)
    parser.add_argument("--num_steps", type=int, default=2000)
    args = parser.parse_args()

    device = get_device("auto")

    # Directory for saving results
    model_name = args.model_path.split("/")[-2]
    out_dir = f"analysis_results/{model_name}"
    os.makedirs(out_dir, exist_ok=True)

    # ScenarioCreator
    sc = ScenarioCreator("src/config/scenario.yaml")

    feature_dict: Dict[str, np.ndarray] = {}

    print("[ScenarioCreator] All environments validated as fixed-size:", sc.get_env_id)

    # loop over difficulties
    for diff in args.difficulties:
        print(f"===== Generating features for: {diff} =====")

        env = sc.create_env(diff)
        sample_obs, _ = env.reset()
        sample_obs = np.array(sample_obs, dtype=np.float32)
        act_dim = env.action_space.n

        policy = load_policy(args.model_path, sample_obs, act_dim, device)

        feats = collect_features(env, policy, device, num_steps=args.num_steps)
        feature_dict[diff] = feats

    # compute metrics
    results = compare_task_feature_dict(feature_dict)

    print("\n====== TASK DISTRIBUTION METRICS ======\n")
    for (a, b), metrics in results.items():
        print(f"{a} vs {b}")
        for k, v in metrics.items():
            print(f"   {k:20s}: {v:.6f}")
        print()

        # Save KDE plot
        plot_kde(feature_dict[a], feature_dict[b],
                 a, b, f"{out_dir}/{a}_vs_{b}_kde.png")

        # Save mean-std plot
        plot_mean_std(feature_dict[a], feature_dict[b],
                      a, b, f"{out_dir}/{a}_vs_{b}_mean.png")

    print(f"\nSaved all figures to: {out_dir}/")


if __name__ == "__main__":
    main()
