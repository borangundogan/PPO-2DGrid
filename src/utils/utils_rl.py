# src/utils_rl.py
import numpy as np
import torch

def compute_gae_standard(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """
    Stateless GAE calculator. Works for both PPO and FOMAML.
    Input: Numpy arrays
    Output: GAE (Advantage) and Returns
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        next_val = last_value if t == T - 1 else values[t + 1]
        
        delta = rewards[t] + gamma * next_val * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv[t] = gae
        
    returns = values + adv
    return adv, returns