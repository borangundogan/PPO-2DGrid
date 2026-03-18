import os
import glob
import argparse
import numpy as np
import torch

from src.scenario_creator.scenario_creator import ScenarioCreator
from src.actor_critic import MLPActorCritic, CNNActorCritic
from src.utils.utils import get_device

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", type=str, default="mediumhard")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--tasks", type=int, default=50)
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
    
    # UYUŞMAZLIK YAMASI: Eski mimariyi yeni mimariye çevir (Weight Mapping)
    if use_cnn and any("feature_extractor" in k for k in state_dict.keys()):
        print(f"[*] Warning: Legacy model architecture detected in {model_path}. Mapping weights...")
        new_state_dict = {}
        for k, v in state_dict.items():
            if "feature_extractor.conv" in k:
                # Eski conv ağırlıklarını hem actor hem critic extractor'a kopyala
                new_k_actor = k.replace("feature_extractor.conv", "actor_extractor.network")
                new_k_critic = k.replace("feature_extractor.conv", "critic_extractor.network")
                new_state_dict[new_k_actor] = v.clone()
                new_state_dict[new_k_critic] = v.clone()
            else:
                new_state_dict[k] = v
        policy.load_state_dict(new_state_dict, strict=False)
    else:
        # Güncel modeller için normal yükleme
        policy.load_state_dict(state_dict)
        
    policy.eval()
    return policy, use_cnn

def process_obs(obs, use_cnn, device):
    if not isinstance(obs, np.ndarray):
        obs = np.array(obs, dtype=np.float32)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    return obs_t.unsqueeze(0) if use_cnn else obs_t.view(1, -1)

def evaluate_model(env, policy, use_cnn, device, seeds):
    total_rews, total_steps = [], []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        done = False
        ep_rew, ep_steps = 0.0, 0
        
        while not done:
            obs_t = process_obs(obs, use_cnn, device)
            with torch.no_grad():
                action, _, _ = policy.act(obs_t, deterministic=True)
            
            obs, reward, terminated, truncated, _ = env.step(action.item())
            ep_rew += reward
            ep_steps += 1
            done = terminated or truncated
            
        total_rews.append(ep_rew)
        total_steps.append(ep_steps)
        
    return np.mean(total_rews), np.mean(total_steps)

def main():
    args = parse_args()
    device = get_device("auto")
    sc = ScenarioCreator(args.config)
    
    model_paths = glob.glob(os.path.join(args.model_dir, "*.pth"))
    if not model_paths:
        print(f"[*] No .pth files found in {args.model_dir}")
        return
        
    test_seeds = [int(s) for s in range(200000, 200000 + args.tasks)]
    env = sc.create_env(args.difficulty)
    
    results = []
    print(f"[*] Initiating Zero-Shot Sweep on {len(model_paths)} checkpoints...")
    print(f"[*] Fixed Evaluation Tasks: {args.tasks}")
    print("-" * 60)
    
    for mp in model_paths:
        policy, use_cnn = load_policy(mp, env, device)
        avg_r, avg_s = evaluate_model(env, policy, use_cnn, device, test_seeds)
        results.append((mp, avg_r, avg_s))
        print(f"Processed: {os.path.basename(mp):<25} | R: {avg_r:.3f} | S: {avg_s:.1f}")
        
    env.close()
    results.sort(key=lambda x: x[1], reverse=True)
    
    print("\n" + "=" * 60)
    print(f"{'RANK':<5} | {'CHECKPOINT':<25} | {'REWARD':<8} | {'STEPS'}")
    print("=" * 60)
    for rank, (mp, r, s) in enumerate(results, 1):
        print(f"#{rank:<4} | {os.path.basename(mp):<25} | {r:<8.3f} | {s:.1f}")
    print("=" * 60)

if __name__ == "__main__":
    main()