import numpy as np
import torch

def evaluate_episode(env, policy, device, max_steps=256, deterministic=False):
    obs, _ = env.reset()    
    total_reward = 0
    done = False
    steps = 0
    
    while not done and steps < max_steps:
        obs_np = np.array(obs)
        if obs_np.ndim == 3:
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).view(1, -1) 
            
        with torch.no_grad():
            action, _, _ = policy.act(obs_t, deterministic=deterministic, epsilon=0.1)
            
        obs, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += reward
        done = terminated or truncated
        steps += 1
        
    return total_reward