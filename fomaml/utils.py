# fomaml/utils.py

import numpy as np
import torch

import numpy as np
import torch

def evaluate_episode(env, policy, device, max_steps=100, deterministic=False):
    obs, _ = env.reset()    
    total_reward = 0
    done = False
    steps = 0
    
    position_history = []
    
    while not done and steps < max_steps:
        obs_np = np.array(obs)
        if obs_np.ndim == 3:
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).view(1, -1) 
            
        current_pos = tuple(env.unwrapped.agent_pos)
        position_history.append(current_pos)
        
        if len(position_history) > 20:
            position_history.pop(0)
            
        is_stuck = len(position_history) == 20 and len(set(position_history)) <= 2
            
        with torch.no_grad():
            if is_stuck:
                action, _, _ = policy.act(obs_t, deterministic=False, epsilon = 0.1)
                position_history.clear()
            else:
                action, _, _ = policy.act(obs_t, deterministic=deterministic)
            
        obs, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += reward
        done = terminated or truncated
        steps += 1
        
    return total_reward