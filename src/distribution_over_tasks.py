import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import gc

from src.scenario_creator.scenario_creator import ScenarioCreator
from src.actor_critic import MLPActorCritic, CNNActorCritic
from src.utils.utils import get_device
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
    parser.add_argument("--difficulty", type=str, default="mediumhard")
    parser.add_argument("--num_tasks", type=int, default=500)
    parser.add_argument("--ppo_model", type=str, required=True)
    parser.add_argument("--fomaml_model", type=str, required=True)
    parser.add_argument("--adapt_steps", type=int, default=0)
    parser.add_argument("--lr_inner", type=float, default=0.01)
    parser.add_argument("--k_support", type=int, default=256)
    parser.add_argument("--config", type=str, default="src/config/scenario.yaml")
    parser.add_argument("--out_dir", type=str, default="eval_results")
    parser.add_argument("--base_seed", type=int, default=100000)
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

def evaluate_zero_shot(env, policy, seed, use_cnn, device):
    obs, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    steps = 0
    
    obs_buf, act_buf, rew_buf, val_buf, logp_buf = [], [], [], [], []

    while not done:
        current_obs = np.array(obs, copy=True, dtype=np.float32)
        obs_t = process_obs(current_obs, use_cnn, device)
        with torch.no_grad():
            action, logp, value = policy.act(obs_t, deterministic=True)
        
        obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        
        obs_buf.append(current_obs)
        act_buf.append(action.item())
        rew_buf.append(reward)
        val_buf.append(value.item())
        logp_buf.append(logp.item())
        
        total_reward += reward
        steps += 1

    with torch.no_grad():
        obs_t = process_obs(obs, use_cnn, device)
        _, _, last_val_tensor = policy.act(obs_t)
        last_val = last_val_tensor.item()
        
        adv_list = []
        gae = 0.0
        for t in reversed(range(len(rew_buf))):
            mask = 0.0 if t == len(rew_buf) - 1 else 1.0
            next_v = last_val if t == len(rew_buf) - 1 else val_buf[t + 1]
            delta = rew_buf[t] + 0.995 * next_v * mask - val_buf[t]
            gae = delta + 0.995 * 0.95 * mask * gae
            adv_list.append(gae)
            
        adv = torch.tensor(adv_list[::-1], dtype=torch.float32, device=device)
        if len(adv) > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        else:
            adv = torch.zeros_like(adv)

        vals = torch.tensor(val_buf, dtype=torch.float32, device=device)
        ret_t = vals + adv
        
        obs_cat = torch.tensor(np.array(obs_buf), dtype=torch.float32, device=device)
        if not use_cnn:
            obs_cat = obs_cat.view(len(obs_buf), -1)
        act_cat = torch.tensor(act_buf, dtype=torch.int64, device=device)
        
        new_logp, _, new_vals = policy.evaluate(obs_cat, act_cat)
        v_loss = ((new_vals - ret_t) ** 2).mean()
        loss = -new_logp.mean() + 0.5 * v_loss
        loss_val = loss.item()

    return total_reward, steps, loss_val

def evaluate_few_shot(env, fast_policy, meta_policy, seed, use_cnn, device, lr_inner, k_support, adapt_steps):
    for param, target_param in zip(fast_policy.parameters(), meta_policy.parameters()):
        param.data.copy_(target_param.data)
        
    fast_policy.train()
    inner_optim = torch.optim.SGD(fast_policy.parameters(), lr=lr_inner)

    for _ in range(adapt_steps):
        obs, _ = env.reset(seed=seed)
        obs_buf, act_buf, rew_buf, val_buf, logp_buf, done_buf = [], [], [], [], [], []

        for _ in range(k_support):
            current_obs = np.array(obs, copy=True, dtype=np.float32)
            obs_t = process_obs(current_obs, use_cnn, device)
            
            with torch.no_grad():
                action, logp, value = fast_policy.act(obs_t, deterministic=False)
            
            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            obs_buf.append(current_obs)
            act_buf.append(action.item())
            rew_buf.append(reward)
            val_buf.append(value.item())
            logp_buf.append(logp.item())
            done_buf.append(done)

            if done:
                obs, _ = env.reset(seed=seed)

        obs_t = process_obs(obs, use_cnn, device)
        with torch.no_grad():
            _, _, last_val_tensor = fast_policy.act(obs_t)
            last_val = last_val_tensor.item()

        adv_list = []
        gae = 0.0
        for t in reversed(range(len(rew_buf))):
            mask = 1.0 - float(done_buf[t])
            next_v = last_val if t == len(rew_buf) - 1 else val_buf[t + 1]
            delta = rew_buf[t] + 0.995 * next_v * mask - val_buf[t]
            gae = delta + 0.995 * 0.95 * mask * gae
            adv_list.append(gae)

        adv = torch.tensor(adv_list[::-1], dtype=torch.float32, device=device)
        if len(adv) > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        else:
            adv = torch.zeros_like(adv)

        vals = torch.tensor(val_buf, dtype=torch.float32, device=device)
        ret_t = (vals + adv).detach()
        old_logp = torch.tensor(logp_buf, dtype=torch.float32, device=device)

        obs_cat = torch.tensor(np.array(obs_buf), dtype=torch.float32, device=device)
        if not use_cnn:
            obs_cat = obs_cat.view(len(obs_buf), -1)
        act_cat = torch.tensor(act_buf, dtype=torch.int64, device=device)

        new_logp, entropy, new_vals = fast_policy.evaluate(obs_cat, act_cat)
        ratio = torch.exp(new_logp - old_logp)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * adv
        pi_loss = -torch.min(surr1, surr2).mean()
        v_loss = ((new_vals - ret_t) ** 2).mean()
        loss = pi_loss + 0.5 * v_loss - 0.05 * entropy.mean()

        inner_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fast_policy.parameters(), max_norm=0.5)
        inner_optim.step()

    fast_policy.eval()
    result = evaluate_zero_shot(env, fast_policy, seed, use_cnn, device)
    
    fast_policy.zero_grad(set_to_none=True)
    return result

def plot_histograms(ppo_data, fomaml_data, metric_name, out_path, total_tasks, title_suffix):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ppo_clean = [x for x in ppo_data if not np.isnan(x)]
    fomaml_clean = [x for x in fomaml_data if not np.isnan(x)]
    
    if not ppo_clean or not fomaml_clean:
        plt.close(fig)
        return

    bins = np.histogram_bin_edges(ppo_clean + fomaml_clean, bins=50)
    
    ax.hist(ppo_clean, bins=bins, alpha=0.5, color='#fc8d62', label='BASE (PPO)')
    ax.hist(fomaml_clean, bins=bins, alpha=0.7, color='#8da0cb', label='FOMAML')
    
    ax.set_xlabel(metric_name, fontsize=14, fontweight='bold')
    ax.set_ylabel("Number of Tasks", fontsize=14, fontweight='bold')
    ax.set_title(f"Distribution of {metric_name} ({title_suffix})", fontsize=16, fontweight='bold', pad=15)
    
    ax.set_ylim(0, total_tasks)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(axis='y', alpha=0.4, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close(fig)

def main():
    args = parse_args()
    device = get_device("auto")
    sc = ScenarioCreator(args.config)
    
    ppo_name = os.path.splitext(os.path.basename(args.ppo_model))[0]
    fomaml_name = os.path.splitext(os.path.basename(args.fomaml_model))[0]
    
    shot_folder = "zero_shot" if args.adapt_steps == 0 else f"{args.adapt_steps}_shot"
    dynamic_out_dir = os.path.join(
        args.out_dir, 
        f"{ppo_name}_vs_{fomaml_name}",
        f"seed_{args.base_seed}",
        shot_folder
    )
    os.makedirs(dynamic_out_dir, exist_ok=True)
    
    test_seeds = [int(s) for s in range(args.base_seed, args.base_seed + args.num_tasks)]
    
    env = sc.create_env(args.difficulty)
    ppo_policy, ppo_cnn = load_policy(args.ppo_model, env, device)
    fomaml_policy, fomaml_cnn = load_policy(args.fomaml_model, env, device)

    obs_sample, _ = env.reset()
    act_dim = env.action_space.n
    if fomaml_cnn:
        fast_policy = CNNActorCritic(obs_sample.shape, act_dim).to(device)
    else:
        fast_policy = MLPActorCritic(int(np.prod(obs_sample.shape)), act_dim).to(device)

    ppo_rews, ppo_steps, ppo_losses = [], [], []
    fomaml_rews, fomaml_steps, fomaml_losses = [], [], []

    print(f"[*] Evaluation | {ppo_name} vs {fomaml_name} | {shot_folder.replace('_', '-').upper()}")
    print(f"[*] Saving to: {dynamic_out_dir}\n")
    
    shot_title = "Zero-Shot" if args.adapt_steps == 0 else f"{args.adapt_steps}-Shot"
    
    total_start_time = time.time()
    batch_start_time = time.time()
    
    for i, seed in enumerate(test_seeds):
        pr, ps, ploss = evaluate_zero_shot(env, ppo_policy, seed, ppo_cnn, device)
        ppo_rews.append(pr)
        ppo_steps.append(ps)
        ppo_losses.append(ploss)
        
        if args.adapt_steps == 0:
            fr, fs, floss = evaluate_zero_shot(env, fomaml_policy, seed, fomaml_cnn, device)
        else:
            fr, fs, floss = evaluate_few_shot(env, fast_policy, fomaml_policy, seed, fomaml_cnn, device, args.lr_inner, args.k_support, args.adapt_steps)
            
        fomaml_rews.append(fr)
        fomaml_steps.append(fs)
        fomaml_losses.append(floss)
        
        if (i + 1) % 10 == 0:
            current_tasks = i + 1
            batch_elapsed = time.time() - batch_start_time
            total_elapsed = time.time() - total_start_time
            
            print(f"Processed {current_tasks}/{args.num_tasks} tasks. | Batch Time: {batch_elapsed:.2f}s | Total Time: {total_elapsed:.2f}s")
            
            gc.collect()
            
            if device.type == 'mps':
                torch.mps.empty_cache()
            elif device.type == 'cuda':
                torch.cuda.empty_cache()

            if current_tasks < args.num_tasks:
                env.close()
                env = sc.create_env(args.difficulty)

            batch_start_time = time.time()

    env.close()

    plot_histograms(ppo_rews, fomaml_rews, "Reward", os.path.join(dynamic_out_dir, "reward_dist.png"), args.num_tasks, shot_title)
    plot_histograms(ppo_steps, fomaml_steps, "Steps to Goal", os.path.join(dynamic_out_dir, "steps_dist.png"), args.num_tasks, shot_title)
    plot_histograms(ppo_losses, fomaml_losses, "Validation Loss", os.path.join(dynamic_out_dir, "loss_dist.png"), args.num_tasks, shot_title)
    
    print(f"\n[*] Calculating Pairwise Statistics (PPO vs FOMAML)...")
    try:
        metrics = compare_two_feature_sets(np.array(ppo_rews).reshape(-1, 1), np.array(fomaml_rews).reshape(-1, 1))
        print(f"--- Reward Distribution Shift ({shot_title}) ---")
        for k, v in metrics.items():
            print(f"    {k:<20}: {v:.6f}")
    except Exception as e:
         pass
         
    final_time = time.time() - total_start_time
    print(f"\n[*] Complete in {final_time:.2f}s. PPO Avg Rew: {np.mean(ppo_rews):.3f} | FOMAML Avg Rew: {np.mean(fomaml_rews):.3f}")

if __name__ == "__main__":
    main()