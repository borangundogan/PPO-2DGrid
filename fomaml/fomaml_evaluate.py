import sys
import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scenario_creator.scenario_creator import ScenarioCreator
from src.utils.utils import get_device, set_seed
from src.fomaml import FOMAML 

plt.rcParams.update({
    "font.family": "serif",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.5,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "black"
})

def process_obs(obs, use_cnn, device):
    if not isinstance(obs, np.ndarray):
        obs = np.array(obs, dtype=np.float32)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    return obs_t.unsqueeze(0) if use_cnn else obs_t.view(1, -1)

def evaluate_policy(env, policy, use_cnn, device, seed):
    obs, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        obs_t = process_obs(obs, use_cnn, device)
        with torch.no_grad():
            action, _, _ = policy.act(obs_t, deterministic=True)
        
        obs, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += reward
        steps += 1
        done = terminated or truncated

    return total_reward, steps

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Meta-RL Adaptation (Pre vs Post)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--difficulty", type=str, default="mediumhard")
    parser.add_argument("--num_tasks", type=int, default=50, help="Number of unique maps to test")
    parser.add_argument("--k_support", type=int, default=256, help="Steps for adaptation (Inner Loop)")
    parser.add_argument("--lr_inner", type=float, default=0.01, help="Inner loop learning rate")
    parser.add_argument("--seed", type=int, default=1000, help="Starting seed for evaluation")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device("auto")
    sc = ScenarioCreator("src/config/scenario.yaml")

    path_parts = os.path.normpath(args.model_path).split(os.sep)
    if "checkpoints" in path_parts:
        try:
            idx = path_parts.index("checkpoints")
            exp_name = path_parts[idx + 1] 
            seed_name = path_parts[idx + 2]
            out_dir = os.path.join("analysis_results", "meta_eval", exp_name, seed_name)
        except IndexError:
            out_dir = os.path.join("analysis_results", "meta_eval", "custom_eval")
    else:
        out_dir = os.path.join("analysis_results", "meta_eval", "custom_eval")
        
    os.makedirs(out_dir, exist_ok=True)
    print(f"[*] Meta-Eval Output Directory: {out_dir}")

    fomaml_helper = FOMAML(
        sc, 
        lr_inner=args.lr_inner, 
        device=device, 
        difficulty=args.difficulty
    )
    
    print(f"[*] Loading meta-model: {os.path.basename(args.model_path)}")
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    
    if fomaml_helper.use_cnn and any("feature_extractor" in k for k in state_dict.keys()):
        new_state_dict = {}
        for k, v in state_dict.items():
            if "feature_extractor.conv" in k:
                new_state_dict[k.replace("feature_extractor.conv", "actor_extractor.network")] = v.clone()
                new_state_dict[k.replace("feature_extractor.conv", "critic_extractor.network")] = v.clone()
            else:
                new_state_dict[k] = v
        fomaml_helper.meta_policy.load_state_dict(new_state_dict, strict=False)
    else:
        fomaml_helper.meta_policy.load_state_dict(state_dict)
        
    fomaml_helper.meta_policy.eval()

    pre_update_rewards, pre_update_steps = [], []
    post_update_rewards, post_update_steps = [], []
    
    print(f"\n[*] Testing {args.num_tasks} unseen tasks (K-Support={args.k_support}, LR={args.lr_inner})")
    print(f"{'Task Seed':<10} | {'Pre-Reward':<12} | {'Post-Reward':<12} | {'Rew-Delta':<10} | {'Steps-Delta'}")
    print("-" * 65)

    for i in range(args.num_tasks):
        task_seed = args.seed + i
        env = sc.create_env(args.difficulty, seed=task_seed)
        
        r_pre, s_pre = evaluate_policy(env, fomaml_helper.meta_policy, fomaml_helper.use_cnn, device, seed=task_seed) 
        
        fast_policy = deepcopy(fomaml_helper.meta_policy)
        fast_policy.train() 
        inner_optim = optim.SGD(fast_policy.parameters(), lr=args.lr_inner)
        
        support_data = fomaml_helper.collect_trajectory(
            env, 
            fast_policy, 
            steps=args.k_support,
            task_seed=task_seed
        )

        loss, _ = fomaml_helper.compute_loss(support_data, fast_policy)
        inner_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fast_policy.parameters(), max_norm=0.5)
        inner_optim.step()
        
        fast_policy.eval() 
        r_post, s_post = evaluate_policy(env, fast_policy, fomaml_helper.use_cnn, device, seed=task_seed)
        
        pre_update_rewards.append(r_pre)
        post_update_rewards.append(r_post)
        pre_update_steps.append(s_pre)
        post_update_steps.append(s_post)
        
        print(f"{task_seed:<10} | {r_pre:<12.3f} | {r_post:<12.3f} | {r_post - r_pre:<10.3f} | {s_post - s_pre:+.1f}")
        env.close()

    plt.figure(figsize=(7, 7))
    plt.scatter(pre_update_rewards, post_update_rewards, alpha=0.7, color='#8da0cb', edgecolors='k', s=60)
    
    max_val = max(1.0, max(post_update_rewards + pre_update_rewards))
    plt.plot([0, max_val], [0, max_val], 'r--', label="No Change", linewidth=2)
    
    plt.title(f"Adaptation Analysis: {args.difficulty.capitalize()} (K={args.k_support})", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Pre-Update Reward (Zero-Shot)", fontsize=12, fontweight='bold')
    plt.ylabel("Post-Update Reward (Few-Shot)", fontsize=12, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "adaptation_scatter.png"), dpi=600)
    plt.close()

    avg_pre, std_pre = np.mean(pre_update_rewards), np.std(pre_update_rewards)
    avg_post, std_post = np.mean(post_update_rewards), np.std(post_update_rewards)
    
    plt.figure(figsize=(6, 6))
    bars = plt.bar(["Pre-Update", "Post-Update"], [avg_pre, avg_post], 
                   yerr=[std_pre, std_post], capsize=8, 
                   color=["#fc8d62", "#8da0cb"], alpha=0.9, edgecolor='black', linewidth=1.2)
    
    plt.title(f"Average Performance Improvement", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("Average Return", fontsize=12, fontweight='bold')
    plt.ylim(0, 1.05) 
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.3f}", 
                 ha='center', va='bottom', fontweight='bold', fontsize=11)
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "adaptation_bar_chart.png"), dpi=600)
    plt.close()

    print(f"\n[*] Results Summary")
    print(f"    Avg Reward Improvement: {avg_post - avg_pre:+.3f}")
    print(f"    Avg Steps Saved       : {np.mean(pre_update_steps) - np.mean(post_update_steps):+.1f}")
    print(f"[*] Ultra-HD plots saved to: {out_dir}/")

if __name__ == "__main__":
    main()