import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from copy import deepcopy
from typing import Dict

from src.scenario_creator.scenario_creator import ScenarioCreator
from src.metrics.task_metrics import compare_two_feature_sets
from src.utils.utils import get_device, set_seed
from src.fomaml import FOMAML 
from src.actor_critic import CNNActorCritic, MLPActorCritic

from utils import evaluate_episode


# --- CORE: Collect Meta-RL Rewards (With Safety Lock) ---
def collect_meta_rewards(sc, difficulty, meta_policy, device, num_tasks=50, start_seed=1000, k_support=40, lr_inner=0.001):
    rewards = []
    
    # Init Helper
    fomaml_helper = FOMAML(sc, device=device, difficulty=difficulty)
    
    SUCCESS_THRESHOLD = 0.60 # Safety Lock Threshold

    for i in range(num_tasks):
        task_seed = start_seed + i
        
        # 1. Pre-Update Eval
        env = sc.create_env(difficulty, seed=task_seed)
        r_pre = evaluate_episode(env, meta_policy, device, deterministic=True)
        
        final_reward = r_pre # Default assumption
        
        # 2. Safety Lock Logic
        if r_pre < SUCCESS_THRESHOLD:
            # Need Adaptation!
            env.reset(seed=task_seed)
            
            # Create Fast Weights
            fast_policy = deepcopy(meta_policy)
            fast_policy.train()
            inner_optim = optim.SGD(fast_policy.parameters(), lr=lr_inner)
            
            # Collect Support Data (Stochastic)
            support_data = fomaml_helper.collect_trajectory(env, meta_policy, steps=k_support)
            
            if support_data["rew"].sum().item() > 0.0:
                loss = fomaml_helper.compute_loss(support_data, fast_policy)
                inner_optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fast_policy.parameters(), max_norm=0.5)
                inner_optim.step()
                
                # 3. Post-Update Eval
                env.reset(seed=task_seed)
                fast_policy.eval()
                final_reward = evaluate_episode(env, fast_policy, device, deterministic=True)
            else:
                # Adaptation failed to find goal, keep Pre-Reward
                pass
        
        rewards.append(final_reward)
        env.close()

    return np.array(rewards)

# --- PLOTTING FUNCTIONS ---
def plot_reward_distribution(r1, r2, name1, name2, save_path):
    plt.figure(figsize=(10, 6))
    data_df = pd.concat([
        pd.DataFrame({"Episode Return": r1, "Task": name1}),
        pd.DataFrame({"Episode Return": r2, "Task": name2})
    ])
    sns.histplot(
        data=data_df, x="Episode Return", hue="Task", stat="probability",  
        kde=False, bins=20, binrange=(-1, 1.1), multiple="layer",    
        palette=["#1f77b4", "#ff7f0e"], alpha=0.5, common_norm=False
    )
    plt.title(f"Meta-Adaptation Performance: {name1} vs {name2}", fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6) 
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_reward_bars(reward_dict, save_path):
    tasks = list(reward_dict.keys())
    means = [np.mean(reward_dict[t]) for t in tasks]
    stds = [np.std(reward_dict[t]) for t in tasks]

    plt.figure(figsize=(7, 5))
    # Add dynamic colors based on number of tasks
    colors = sns.color_palette("husl", len(tasks))
    bars = plt.bar(tasks, means, yerr=stds, capsize=6, color=colors)
    plt.title("Mean Meta-Adapted Reward per Task")
    plt.ylabel("Mean Return")
    plt.ylim(-1.0, 1.1)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, round(yval, 2), ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze FOMAML Distribution Shift")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--difficulties", nargs="+", required=True, help="e.g. mediumhard hard")
    parser.add_argument("--num_tasks", type=int, default=50)
    parser.add_argument("--seed", type=int, default=4382)
    parser.add_argument("--k_support", type=int, default=40)
    parser.add_argument("--lr_inner", type=float, default=0.001)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device("auto")
    sc = ScenarioCreator("src/config/scenario.yaml")

    # Load Base Meta-Policy
    dummy_env = sc.create_env(args.difficulties[0], seed=42)
    obs_sample, _ = dummy_env.reset()
    act_dim = dummy_env.action_space.n
    obs_sample = np.array(obs_sample)
    
    if obs_sample.ndim == 3:
        meta_policy = CNNActorCritic(obs_sample.shape, act_dim).to(device)
    else:
        meta_policy = MLPActorCritic(int(np.prod(obs_sample.shape)), act_dim).to(device)
        
    print(f"[Analysis] Loading Meta-Model: {args.model_path}")
    meta_policy.load_state_dict(torch.load(args.model_path, map_location=device))
    meta_policy.eval()

    # --- UPDATED OUTPUT DIR LOGIC ---
    norm_path = os.path.normpath(args.model_path)
    path_parts = norm_path.split(os.sep)
    
    if "checkpoints" in path_parts:
        try:
            idx = path_parts.index("checkpoints")
            exp_name = path_parts[idx + 1] # e.g. MERLIN-Mediumhard...
            # OLD: seed_name = path_parts[idx + 2] 
            # NEW: Use the seed from arguments, not the training seed
            out_dir = os.path.join("analysis_results", exp_name, f"seed_{args.seed}")
        except IndexError:
             out_dir = os.path.join("analysis_results", "custom_analysis", f"seed_{args.seed}")
    else:
        out_dir = os.path.join("analysis_results", "custom_analysis", f"seed_{args.seed}")

    os.makedirs(out_dir, exist_ok=True)
    print(f"[Analysis] Saving results to: {out_dir}")

    reward_dict = {}

    # --- MAIN LOOP ---
    for diff in args.difficulties:
        print(f"\n[Collecting] Meta-Adaptation stats for: {diff}...")
        
        rewards = collect_meta_rewards(
            sc, diff, meta_policy, device,
            num_tasks=args.num_tasks,
            start_seed=args.seed,
            k_support=args.k_support,
            lr_inner=args.lr_inner
        )
        
        reward_dict[diff] = rewards
        print(f"   -> Mean Reward: {np.mean(rewards):.3f}")

    # --- PLOTTING ---
    print(f"\n[Plotting] Generating charts...")
    plot_reward_bars(reward_dict, f"{out_dir}/meta_reward_bar_chart.png")

    keys = list(reward_dict.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            plot_reward_distribution(reward_dict[a], reward_dict[b], a, b, f"{out_dir}/{a}_vs_{b}_dist.png")
            
            # Metrics
            metrics = compare_two_feature_sets(reward_dict[a].reshape(-1, 1), reward_dict[b].reshape(-1, 1))
            print(f"\nMetrics ({a} vs {b}):")
            for k, v in metrics.items():
                print(f"   {k}: {v:.4f}")
    
    print(f"\n[Done] All plots saved to: {out_dir}")

if __name__ == "__main__":
    main()