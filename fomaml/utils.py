# fomaml/utils.py

import numpy as np
import torch

def evaluate_episode(env, policy, device, max_steps=100, deterministic=False):
    """
    Runs a deterministic evaluation episode (Argmax action).
    Returns: Total Reward
    """
    obs, _ = env.reset()    
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
            # Deterministic = True for evaluation
            action, _, _ = policy.act(obs_t, deterministic=deterministic)
            
        obs, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += reward
        done = terminated or truncated
        steps += 1
        
    return total_reward