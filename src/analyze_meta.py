# analyze_meta.py

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from collections import defaultdict

from src.meta.task_sampler import MetaTaskSampler
from src.meta.maml_vpg import MAMLVPG
from src.meta.meta_utils import collect_episodes, fast_adapt, compute_policy_loss
from src.utils import get_device, set_seed

from src.metrics.task_metrics import compare_two_feature_sets

def parse_args():
    parser = argparse.ArgumentParser("Quantitative Meta-RL Analysis")
    parser.add_argument("--config", type=str, default="src/config/scenario.yaml")
    parser.add_argument("--difficulties", nargs="+", default=["easy", "medium", "hard"])
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth checkpoint")
    
    # Analysis Parameters
    parser.add_argument("--num_tasks", type=int, default=50, help="Number of distinct tasks per difficulty")
    parser.add_argument("--n_support", type=int, default=10, help="Episodes for adaptation")
    parser.add_argument("--n_query", type=int, default=10, help="Episodes for evaluation (increased for better statistics)")
    parser.add_argument("--adaptation_steps", type=int, default=1)
    parser.add_argument("--inner_lr", type=float, default=0.1)
    
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def evaluate_policy_raw(env, policy, device, n_episodes):
    """Returns a LIST of rewards for distribution analysis."""
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        obs = np.array(obs, dtype=np.float32)
        done = False
        ep_reward = 0
        
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            if obs.ndim == 3: obs_t = obs_t.unsqueeze(0) / 255.0
            else: obs_t = obs_t.view(1, -1) / 255.0
            
            with torch.no_grad():
                logits = policy.actor(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().item()
            
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
    return rewards

def plot_kde_comparison(pre_rewards, post_rewards, difficulty, save_path):
    """Plots Pre vs Post adaptation reward distributions."""
    plt.figure(figsize=(8, 6))
    sns.kdeplot(pre_rewards, label="Pre-Adaptation (Zero-Shot)", fill=True, color="red", alpha=0.3)
    sns.kdeplot(post_rewards, label="Post-Adaptation (Few-Shot)", fill=True, color="green", alpha=0.3)
    plt.title(f"Adaptation Impact: {difficulty.upper()} Tasks")
    plt.xlabel("Episode Return")
    plt.ylabel("Density")
    plt.xlim(0, 1.01) # Assuming rewards are normalized or bounded ~1.0
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device("auto")
    
    print(f"[Analysis] Loading model: {args.model_path}")

    # Output setup
    os.makedirs("analysis_results", exist_ok=True)
    
    # Dummy env for dimensions
    temp_sampler = MetaTaskSampler(args.config, ["easy"], base_seed=0)
    dummy_env = temp_sampler.sc.create_env("easy", seed=0)
    obs_dim = int(np.prod(dummy_env.observation_space.shape))
    act_dim = dummy_env.action_space.n
    dummy_env.close()

    meta_agent = MAMLVPG(obs_dim, act_dim, args.inner_lr, 0, 0.99, device)
    meta_agent.policy.load_state_dict(torch.load(args.model_path, map_location=device))

    # Data containers
    summary_data = [] # For CSV/Barplot

    dist_data = defaultdict(lambda: defaultdict(list))

    # --- MAIN LOOP ---
    for difficulty in args.difficulties:
        print(f"\n>>> Processing Difficulty: {difficulty.upper()}")
        
        # Test tasks (different seed offset)
        sampler = MetaTaskSampler(args.config, [difficulty], base_seed=args.seed + 2000)
        tasks = sampler.sample(args.num_tasks)
        
        for i, task_dict in enumerate(tasks):
            env = task_dict['env']
            
            # 1. Pre-Adapt Evaluation (Zero-Shot)
            pre_rewards_list = evaluate_policy_raw(env, meta_agent.policy, device, args.n_query)
            pre_mean = np.mean(pre_rewards_list)
            
            # 2. Adaptation (Inner Loop)
            adapted_policy = meta_agent.policy
            for _ in range(args.adaptation_steps):
                trajs = collect_episodes(env, adapted_policy, device, args.n_support, max_steps=200)
                loss = compute_policy_loss(trajs, adapted_policy, 0.99, device)
                grads = torch.autograd.grad(loss, adapted_policy.parameters(), create_graph=False, allow_unused=True)
                adapted_policy = fast_adapt(adapted_policy, grads, args.inner_lr)
            
            # 3. Post-Adapt Evaluation (Few-Shot)
            post_rewards_list = evaluate_policy_raw(env, adapted_policy, device, args.n_query)
            post_mean = np.mean(post_rewards_list)
            
            # Store Summary
            summary_data.append({'Difficulty': difficulty, 'Stage': 'Pre-Adapt', 'Reward': pre_mean})
            summary_data.append({'Difficulty': difficulty, 'Stage': 'Post-Adapt', 'Reward': post_mean})
            
            # Store Raw Distribution Data (Flat list extension)
            dist_data[difficulty]['pre'].extend(pre_rewards_list)
            dist_data[difficulty]['post'].extend(post_rewards_list)
            
            env.close()
            
            if (i+1) % 10 == 0:
                print(f"   Task {i+1}/{args.num_tasks} done.")

        # --- QUANTITATIVE ANALYSIS (PER DIFFICULTY) ---
        print(f"\n--- Metrics for {difficulty.upper()} (Pre vs Post) ---")
        
        r_pre = np.array(dist_data[difficulty]['pre']).reshape(-1, 1)
        r_post = np.array(dist_data[difficulty]['post']).reshape(-1, 1)
        
        # 1. Compute Distances (Wasserstein, JS, etc.)
        metrics = compare_two_feature_sets(r_pre, r_post)
        for k, v in metrics.items():
            print(f"   {k:20s}: {v:.6f}")
            
        # 2. Plot KDE
        kde_path = f"analysis_results/{difficulty}_adaptation_kde.png"
        plot_kde_comparison(
            dist_data[difficulty]['pre'], 
            dist_data[difficulty]['post'], 
            difficulty, 
            kde_path
        )
        print(f"   [Plot] Saved KDE to {kde_path}")

    df = pd.DataFrame(summary_data)
    df.to_csv("analysis_results/meta_summary.csv", index=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Difficulty", y="Reward", hue="Stage", palette="viridis", errorbar="sd", capsize=.1)
    plt.title("Meta-RL Adaptation Performance (Pre vs Post)")
    plt.savefig("analysis_results/meta_summary_bar.png")
    print("\n[Done] Analysis complete. Check 'analysis_results/' folder.")

if __name__ == "__main__":
    import src.meta.meta_utils 
    main()