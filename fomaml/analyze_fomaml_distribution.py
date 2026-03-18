import sys
import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scenario_creator.scenario_creator import ScenarioCreator
from src.metrics.task_metrics import compare_two_feature_sets
from src.utils.utils import get_device, set_seed
from src.fomaml import FOMAML 
from src.actor_critic import CNNActorCritic, MLPActorCritic

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

def evaluate_adapted_policy(env, policy, use_cnn, device, seed):
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

def collect_meta_metrics(sc, difficulty, meta_policy, use_cnn, device, num_tasks=50, start_seed=1000, k_support=40, lr_inner=0.001, adapt_steps=1):
    rewards = []
    steps_list = []
    
    fomaml_helper = FOMAML(sc, device=device, difficulty=difficulty)

    for i in range(num_tasks):
        task_seed = start_seed + i
        env = sc.create_env(difficulty, seed=task_seed)
        
        if adapt_steps == 0:
            final_reward, final_steps = evaluate_adapted_policy(env, meta_policy, use_cnn, device, seed=task_seed)
        else:
            fast_policy = deepcopy(meta_policy)
            fast_policy.train()
            inner_optim = optim.SGD(fast_policy.parameters(), lr=lr_inner)
            
            for _ in range(adapt_steps):
                support_data = fomaml_helper.collect_trajectory(env, fast_policy, steps=k_support, task_seed=task_seed)
                loss, _ = fomaml_helper.compute_loss(support_data, fast_policy)
                inner_optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fast_policy.parameters(), max_norm=0.5)
                inner_optim.step()
                
            fast_policy.eval()
            final_reward, final_steps = evaluate_adapted_policy(env, fast_policy, use_cnn, device, seed=task_seed)
        
        rewards.append(final_reward)
        steps_list.append(final_steps)
        env.close()

    return np.array(rewards), np.array(steps_list)

def plot_generalization(results_dict, metric_idx, metric_name, out_path, color, shot_text):
    difficulties = list(results_dict.keys())
    means = [np.mean(results_dict[d][metric_idx]) for d in difficulties]
    stds = [np.std(results_dict[d][metric_idx]) for d in difficulties]

    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(difficulties))
    
    ax.bar(x_pos, means, yerr=stds, capsize=8, alpha=0.8, color=color, edgecolor='black', linewidth=1.2)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([d.capitalize() for d in difficulties], fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=14, fontweight='bold')
    
    ax.set_title(f"FOMAML {shot_text} Generalization across Difficulties", fontsize=16, fontweight='bold', pad=15)
    
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
        data=data_df, x="Episode Return", hue="Task", stat="probability",  
        kde=False, bins=50, binrange=(0, 1.05), multiple="layer", element="step",   
        palette=["#fc8d62", "#8da0cb"], alpha=0.6, common_norm=False, linewidth=1.2, ax=ax
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
    parser = argparse.ArgumentParser(description="Analyze FOMAML Distribution Shift")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--difficulties", nargs="+", required=True, help="e.g. easy medium mediumhard hard hardest")
    parser.add_argument("--num_tasks", type=int, default=50)
    parser.add_argument("--seed", type=int, default=300000)
    parser.add_argument("--k_support", type=int, default=256)
    parser.add_argument("--lr_inner", type=float, default=0.01)
    parser.add_argument("--adapt_steps", type=int, default=1, help="0 for Zero-Shot, >0 for Few-Shot")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device("auto")
    sc = ScenarioCreator("src/config/scenario.yaml")

    dummy_env = sc.create_env(args.difficulties[0], seed=42)
    obs_sample, _ = dummy_env.reset()
    act_dim = dummy_env.action_space.n
    obs_sample = np.array(obs_sample)
    use_cnn = obs_sample.ndim == 3
    
    if use_cnn:
        meta_policy = CNNActorCritic(obs_sample.shape, act_dim).to(device)
    else:
        meta_policy = MLPActorCritic(int(np.prod(obs_sample.shape)), act_dim).to(device)
        
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    
    if use_cnn and any("feature_extractor" in k for k in state_dict.keys()):
        new_state_dict = {}
        for k, v in state_dict.items():
            if "feature_extractor.conv" in k:
                new_state_dict[k.replace("feature_extractor.conv", "actor_extractor.network")] = v.clone()
                new_state_dict[k.replace("feature_extractor.conv", "critic_extractor.network")] = v.clone()
            else:
                new_state_dict[k] = v
        meta_policy.load_state_dict(new_state_dict, strict=False)
    else:
        meta_policy.load_state_dict(state_dict)
        
    meta_policy.eval()

    # UPDATE: Shot folder structure
    shot_folder = "zero_shot" if args.adapt_steps == 0 else f"{args.adapt_steps}_shot"

    path_parts = os.path.normpath(args.model_path).split(os.sep)
    if "checkpoints" in path_parts:
        try:
            idx = path_parts.index("checkpoints")
            exp_name = path_parts[idx + 1] 
            seed_name = path_parts[idx + 2]
            # UPDATE: Appending the shot_folder
            out_dir = os.path.join("analysis_results", exp_name, seed_name, shot_folder)
        except IndexError:
             out_dir = os.path.join("analysis_results", "fomaml_eval", f"seed_{args.seed}", shot_folder)
    else:
        out_dir = os.path.join("analysis_results", "fomaml_eval", f"seed_{args.seed}", shot_folder)

    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[*] Output Directory: {out_dir}")
    
    mode_str = "Zero-Shot" if args.adapt_steps == 0 else f"{args.adapt_steps}-Shot"
    print(f"[*] Analyzing FOMAML {mode_str} Generalization | K-Support: {args.k_support} | LR: {args.lr_inner}\n")

    results_dict = {}

    for diff in args.difficulties:
        print(f"--- Evaluating on: {diff.upper()} ---")
        rewards, steps = collect_meta_metrics(
            sc, diff, meta_policy, use_cnn, device,
            num_tasks=args.num_tasks,
            start_seed=args.seed,
            k_support=args.k_support,
            lr_inner=args.lr_inner,
            adapt_steps=args.adapt_steps
        )
        
        results_dict[diff] = (rewards, steps)
        print(f"  => Avg Reward: {np.mean(rewards):.3f} | Avg Steps: {np.mean(steps):.1f}\n")

    print(f"[*] Generating OOD Bar Charts...")
    shot_text = "Zero-Shot" if args.adapt_steps == 0 else f"{args.adapt_steps}-Shot"
    
    # We can simplify the filenames now that they are in separate folders
    plot_generalization(results_dict, 0, "Average Reward", os.path.join(out_dir, "reward_generalization.png"), "#8da0cb", shot_text)
    plot_generalization(results_dict, 1, "Average Steps to Goal", os.path.join(out_dir, "steps_generalization.png"), "#fc8d62", shot_text)

    keys = list(results_dict.keys())
    print(f"[*] Generating Cross-Task Distribution Metrics\n")

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            r1, r2 = results_dict[a][0], results_dict[b][0]
            
            plot_path = os.path.join(out_dir, f"dist_shift_{a}_vs_{b}.png")
            plot_reward_distribution(r1, r2, a, b, plot_path)
            
            try:
                metrics = compare_two_feature_sets(r1.reshape(-1, 1), r2.reshape(-1, 1))
                print(f"[{a.upper()} vs {b.upper()}]")
                for k, v in metrics.items():
                    print(f"    {k:<20}: {v:.6f}")
                print()
            except Exception as e:
                print(f"[{a.upper()} vs {b.upper()}] Warning: Metric calculation failed ({e})\n")
    
    print(f"[*] Analysis Complete. All artifacts saved to: {out_dir}/")

if __name__ == "__main__":
    main()