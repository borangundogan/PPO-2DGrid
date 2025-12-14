# src/meta/meta_utils.py

import numpy as np
import torch
from copy import deepcopy
from typing import List, Tuple

Trajectory = Tuple[np.ndarray, np.ndarray, np.ndarray]
# (obs_seq, action_seq, reward_seq)


def fast_adapt(network, grads, lr):
    """
    Creates a functional copy of the network.
    Replaces parameters with (param - lr * grad) tensors to preserve the computational graph.
    """
    cloned_network = deepcopy(network)
    
    # Map parameter names to their computed gradients
    param_grad_map = {name: g for (name, _), g in zip(network.named_parameters(), grads)}
    
    for name, param in list(cloned_network.named_parameters()):
        if name in param_grad_map and param_grad_map[name] is not None:
            grad = param_grad_map[name]
            
            # Get original parameter to maintain graph connection
            orig_param = dict(network.named_parameters())[name]
            
            # Compute new parameter value (θ' = θ - α∇θ)
            new_val = orig_param - lr * grad
            
            # Replace nn.Parameter with the computed Tensor
            module_path = name.split('.')
            sub_module = cloned_network
            for p in module_path[:-1]:
                sub_module = getattr(sub_module, p)
            
            # Remove original parameter and set the new tensor as an attribute
            delattr(sub_module, module_path[-1])
            setattr(sub_module, module_path[-1], new_val)
            
    return cloned_network


def collect_episodes(
    env,
    policy,
    device,
    n_episodes: int,
    max_steps: int = 200,
) -> List[Trajectory]:
    """
    Collect episodes using the given policy in the environment.
    """
    trajectories = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        obs = np.array(obs, dtype=np.float32)

        obs_list = []
        act_list = []
        rew_list = []

        for _ in range(max_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

            if obs.ndim == 3:
                obs_t = obs_t.unsqueeze(0) / 255.0
            else:
                obs_t = obs_t.view(1, -1) / 255.0

            with torch.no_grad():
                logits = policy.actor(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)

            obs_list.append(obs)
            act_list.append(action)
            rew_list.append(reward)

            obs = np.array(next_obs, dtype=np.float32)

            if terminated or truncated:
                break

        trajectories.append((
            np.array(obs_list, dtype=np.float32),
            np.array(act_list, dtype=np.int64),
            np.array(rew_list, dtype=np.float32),
        ))

    return trajectories


def compute_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """
    Compute discounted returns G_t.
    """
    G = 0.0
    returns = []

    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)

    returns.reverse()
    return np.array(returns, dtype=np.float32)


def compute_policy_loss(
    trajectories: List[Trajectory],
    policy,
    gamma: float,
    device,
) -> torch.Tensor:
    """
    REINFORCE loss: L = -E[ G_t * log pi(a_t | s_t) ]
    """
    losses = []

    for obs_arr, act_arr, rew_arr in trajectories:
        returns = compute_returns(rew_arr, gamma)

        obs_t = torch.tensor(obs_arr, dtype=torch.float32, device=device)
        act_t = torch.tensor(act_arr, dtype=torch.long, device=device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=device)

        # Normalize returns for stability
        ret_t = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)

        if obs_arr.ndim == 4:
            obs_t = obs_t / 255.0
        else:
            obs_t = obs_t.view(len(obs_arr), -1) / 255.0

        logits = policy.actor(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(act_t)

        loss = -(logp * ret_t).mean()
        losses.append(loss)

    return torch.stack(losses).mean()