import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as pd
import seaborn as sns
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scenario_creator.scenario_creator import ScenarioCreator
from src.actor_critic import MLPActorCritic, CNNActorCritic
from src.utils.utils import get_device, set_seed
from src.metrics.task_metrics import compare_two_feature_sets

plt.rcParams.update({
    "font.family": "serif",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.5,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "black"
})

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--difficulties", nargs='+', default=["easy", "medium", "mediumhard", "hard", "hardest"])
    parser.add_argument("--num_tasks", type=int, default=50)
    parser.add_argument("--base_seed", type=int, default=300000)
    parser.add_argument("--config", type=str, default="src/config/scenario.yaml")
    return parser.parse_args()

def load_policy(model_path, env, device):
    obs_sample, _ = env.reset()
    act_dim = env.action_space.n
    use_cnn = obs_sample.ndim == 3
    
    if use_cnn:
        policy = CNNActorCritic(obs_sample.shape, act_dim).to(device)
    else:
        policy = MLPActorCritic(int(np.prod(obs_sample.shape)), act_dim).to(device)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    if use_cnn and any("feature_extractor" in k for k in state_dict.keys()):
        new_state_dict = {}
        for k, v in state_dict.items():
            if "feature_extractor.conv" in k:
                new_state_dict[k.replace("feature_extractor.conv", "actor_extractor.network")] = v.clone()
                new_state_dict[k.replace("feature_extractor.conv", "critic_extractor.network")] = v.clone()
            else:
                new_state_dict[k] = v
        policy.load_state_dict(new_state_dict, strict=False)
    else:
        policy.load_state_dict(state_dict)
        
    policy.eval()
    return policy, use_cnn

def process_obs(obs, use_cnn, device):
    if not isinstance(obs, np.ndarray):
        obs = np.array(obs, dtype=np.float32)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    return obs_t.unsqueeze(0) if use_cnn else obs_t.view(1, -1)

def collect_zero_shot_metrics(env, policy, use_cnn, device, num_tasks, base_seed):
    rewards = []
    steps_list = []
    
    for i in range(num_tasks):
        seed = base_seed + i
        obs, _ = env.reset(seed=seed)
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            obs_t = process_obs(obs, use_cnn, device)
            with torch.no_grad():
                action, _, _ = policy.act(obs_t, deterministic=True)
            
            obs, reward, terminated, truncated, _ = env.step(action.item())
            ep_reward += reward
            steps += 1
            done = terminated or truncated

        rewards.append(ep_reward)
        steps_list.append(steps)

    return np.array(rewards), np.array(steps_list)

def plot_generalization(results_dict, metric_idx, metric_name, out_path, color):
    difficulties = list(results_dict.keys())
    means = [np.mean(results_dict[d][metric_idx]) for d in difficulties]
    stds = [np.std(results_dict[d][metric_idx]) for d in difficulties]

    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(difficulties))
    
    ax.bar(x_pos, means, yerr=stds, capsize=8, alpha=0.8, color=color, edgecolor='black', linewidth=1.2)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([d.capitalize() for d in difficulties], fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=14, fontweight='bold')
    ax.set_title(f"PPO Zero-Shot Generalization across Difficulties", fontsize=16, fontweight='bold', pad=15)
    
    if metric_name == "Average Reward":
        ax.set_ylim(0, 1.05)
        
    ax.grid(axis='y', alpha=0.4, linestyle='--')
    
    for i, v in enumerate(means):
        ax.text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close()

def plot_reward_distribution(r1, r2, name1, name2, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    
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
        bins=50,        
        binrange=(0, 1.05),     
        multiple="layer",    
        element="step",      
        palette=["#fc8d62", "#8da0cb"], 
        alpha=0.6,          
        common_norm=False,    
        linewidth=1.2,
        ax=ax
    )
    
    ax.set_ylabel("Probability", fontsize=14, fontweight='bold')
    ax.set_xlabel("Episode Return", fontsize=14, fontweight='bold')
    ax.set_title(f"Distribution Shift: {name1.upper()} vs {name2.upper()}", fontsize=16, fontweight='bold', pad=15)
    
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle='--', alpha=0.4) 
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

def main():
    args = parse_args()
    set_seed(args.base_seed)
    device = get_device("auto")
    sc = ScenarioCreator(args.config)

    path_parts = os.path.normpath(args.model_path).split(os.sep)
    if "checkpoints" in path_parts:
        try:
            idx = path_parts.index("checkpoints")
            exp_name = path_parts[idx + 1]
            seed_name = path_parts[idx + 2]
            out_dir = os.path.join("analysis_results", exp_name, seed_name)
        except IndexError:
            out_dir = os.path.join("analysis_results", "custom_eval", f"seed_{args.base_seed}")
    else:
        out_dir = os.path.join("analysis_results", "custom_eval", f"seed_{args.base_seed}")

    os.makedirs(out_dir, exist_ok=True)
    print(f"[*] Saving results to: {out_dir}")

    results_dict = {}

    dummy_env = sc.create_env(args.difficulties[0])
    policy, use_cnn = load_policy(args.model_path, dummy_env, device)
    dummy_env.close()

    print(f"[*] Starting PPO Generalization & Distribution Analysis")

    for diff in args.difficulties:
        print(f"  -> Collecting trajectories for: {diff.upper()}")
        env = sc.create_env(diff, seed=args.base_seed)
        rewards, steps = collect_zero_shot_metrics(env, policy, use_cnn, device, args.num_tasks, args.base_seed)
        results_dict[diff] = (rewards, steps)
        env.close()

    print(f"\n[*] Generating OOD Bar Charts")
    plot_generalization(results_dict, 0, "Average Reward", os.path.join(out_dir, "ppo_reward_generalization.png"), "#fc8d62")
    plot_generalization(results_dict, 1, "Average Steps to Goal", os.path.join(out_dir, "ppo_steps_generalization.png"), "#8da0cb")

    keys = list(results_dict.keys())
    print(f"\n[*] Generating Cross-Task Distribution Metrics & Plots\n")

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            r1, r2 = results_dict[a][0], results_dict[b][0]

            plot_path = os.path.join(out_dir, f"dist_shift_{a}_vs_{b}.png")
            plot_reward_distribution(r1, r2, a, b, plot_path)

            try:
                metrics = compare_two_feature_sets(r1.reshape(-1, 1), r2.reshape(-1, 1))
                print(f"{a.upper()} vs {b.upper()}")
                for k, v in metrics.items():
                    print(f"    {k:<20}: {v:.6f}")
                print()
            except Exception as e:
                print(f"  [{a.upper()} vs {b.upper()}] Warning: Metric calculation skipped ({e})\n")

    print(f"[*] Analysis Complete. Outputs saved to: {out_dir}/")

if __name__ == "__main__":
    main()