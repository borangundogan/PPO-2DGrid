import argparse
import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from copy import deepcopy

from src.scenario_creator.scenario_creator import ScenarioCreator
from src.utils.utils import get_device, set_seed
from src.fomaml import FOMAML  # Reuse your existing class logic

def evaluate_episode(env, policy, device, max_steps=100, deterministic = False):
    """
    Runs a deterministic evaluation episode (Argmax action).
    Returns: Total Reward
    """
    obs, _ = env.reset()
    # Note: env.reset() is seeded externally before calling this function
    
    total_reward = 0
    done = False
    steps = 0
    
    while not done and steps < max_steps:
        # Convert Obs to Tensor
        obs_np = np.array(obs)
        if obs_np.ndim == 3: # H,W,C
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        else: # Flattened
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).view(1, -1) 
            
        with torch.no_grad():
            # Deterministic = True for for outer loop
            action, _, _ = policy.act(obs_t, deterministic=deterministic)
            
        obs, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += reward
        done = terminated or truncated
        steps += 1
        
    return total_reward

def main():
    parser = argparse.ArgumentParser(description="Evaluate Meta-RL Adaptation (Pre vs Post)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--difficulty", type=str, default="medium")
    parser.add_argument("--num_tasks", type=int, default=20, help="Number of unique maps to test")
    parser.add_argument("--k_support", type=int, default=50, help="Steps for adaptation (Inner Loop)")
    parser.add_argument("--lr_inner", type=float, default=0.01, help="Inner loop learning rate")
    parser.add_argument("--seed", type=int, default=1000, help="Starting seed for evaluation")
    args = parser.parse_args()

    # 1. Setup
    set_seed(args.seed)
    device = get_device("auto")
    sc = ScenarioCreator("src/config/scenario.yaml")

    # Output Directory
    path_parts = args.model_path.split(os.sep)
    if len(path_parts) >= 3:
        exp_name = path_parts[-2] # medium_seed42
    else:
        exp_name = "custom_eval"
        
    out_dir = os.path.join("analysis_results", "meta_eval", exp_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[Meta-Eval] Saving results to: {out_dir}")

    # 2. Load FOMAML Helper
    fomaml_helper = FOMAML(
        sc, 
        lr_inner=args.lr_inner, 
        device=device, 
        difficulty=args.difficulty
    )
    
    # Load the Trained Weights
    print(f"[Meta-Eval] Loading model from {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=device)
    fomaml_helper.meta_policy.load_state_dict(state_dict)
    fomaml_helper.meta_policy.eval()

    # 3. Evaluation Loop
    pre_update_rewards = []
    post_update_rewards = []
    
    print(f"\n[Meta-Eval] Testing on {args.num_tasks} unique tasks...")
    print(f"{'Task Seed':<10} | {'Pre-Reward':<12} | {'Post-Reward':<12} | {'Delta':<10}")
    print("-" * 50)

    for i in range(args.num_tasks):
        task_seed = args.seed + i
        
        # --- A. Pre-Update Evaluation (Zero-Shot) ---
        # Create env with specific seed
        env = sc.create_env(args.difficulty, seed=task_seed)
        
        # Run Evaluation (Deterministic) using the Initialization (Theta)
        # Note: We must reset env manually here inside evaluate_episode
        r_pre = evaluate_episode(env, fomaml_helper.meta_policy, device, False) 
        
        # --- B. Inner Loop Adaptation (One-Shot) ---
        # 1. Reset Env to same seed (Support Set)
        env.reset(seed=task_seed)
        
        # 2. Create Fast Weights (Theta')
        fast_policy = deepcopy(fomaml_helper.meta_policy)
        fast_policy.train() # Enable grad tracking for update
        inner_optim = optim.SGD(fast_policy.parameters(), lr=args.lr_inner)
        
        # 3. Collect Support Trajectory (Stochastic Exploration)
        # Reusing fomaml_helper.collect_trajectory logic
        support_data = fomaml_helper.collect_trajectory(
            env, 
            fomaml_helper.meta_policy, 
            steps=args.k_support
        )
        
        # 4. Compute Loss & Update
        loss = fomaml_helper.compute_loss(support_data, fast_policy)
        inner_optim.zero_grad()
        loss.backward()
        inner_optim.step()
        
        # --- C. Post-Update Evaluation (Few-Shot) ---
        # Reset Env to same seed (Query Set - The Test)
        # Important: Env state is reset, but the *Map configuration* is identical.
        env.reset(seed=task_seed)
        fast_policy.eval() 

        # Switch to Deterministic Mode
        r_post = evaluate_episode(env, fast_policy, device, True)
        
        # Logging
        pre_update_rewards.append(r_pre)
        post_update_rewards.append(r_post)
        
        print(f"{task_seed:<10} | {r_pre:<12.3f} | {r_post:<12.3f} | {r_post - r_pre:+.3f}")
        
        env.close()

    # 4. Visualization
    
    # Plot 1: Scatter Plot (Adaptation Delta)
    plt.figure(figsize=(7, 7))
    plt.scatter(pre_update_rewards, post_update_rewards, alpha=0.7, color='blue', edgecolors='k')
    
    # Diagonal line (No improvement)
    max_val = max(1.0, max(post_update_rewards + pre_update_rewards))
    plt.plot([0, max_val], [0, max_val], 'r--', label="No Change")
    
    plt.title(f"Adaptation Analysis: {args.difficulty} (K={args.k_support})")
    plt.xlabel("Pre-Update Reward (Zero-Shot)")
    plt.ylabel("Post-Update Reward (Few-Shot)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "adaptation_scatter.png"))
    plt.close()

    # Plot 2: Bar Comparison
    avg_pre = np.mean(pre_update_rewards)
    avg_post = np.mean(post_update_rewards)
    std_pre = np.std(pre_update_rewards)
    std_post = np.std(post_update_rewards)
    
    plt.figure(figsize=(6, 5))
    bars = plt.bar(["Pre-Update", "Post-Update"], [avg_pre, avg_post], 
                   yerr=[std_pre, std_post], capsize=10, 
                   color=["#d62728", "#2ca02c"], alpha=0.8)
    
    plt.title(f"Average Performance Improvement (N={args.num_tasks})")
    plt.ylabel("Average Return")
    plt.ylim(0, 1.1)
    
    # Add values on top
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 3), 
                 ha='center', va='bottom', fontweight='bold')
        
    plt.savefig(os.path.join(out_dir, "adaptation_bar_chart.png"))
    plt.close()

    print(f"\n[Results] Average Improvement: {avg_post - avg_pre:+.4f}")
    print(f"[Results] Plots saved to: {out_dir}")

if __name__ == "__main__":
    main()