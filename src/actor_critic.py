import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.utils_rl import layer_init

# --- CNN Feature Extractor (Optimized with Stride) ---
class CNNFeatureExtractor(nn.Module):
    def __init__(self, obs_shape, input_channels=3):
        """
        obs_shape: (H, W, C) coming directly from env reset().
        """
        super().__init__()

        # NatureCNN-like structure: Aggressive downsampling using stride
        # This reduces the flattened size significantly (e.g. 7x7 output instead of 84x84)
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # DYNAMIC FEATURE SIZE COMPUTATION
        with torch.no_grad():
            # Dummy input: (B, C, H, W)
            dummy = torch.zeros(
                1,
                input_channels,
                obs_shape[0],  # H
                obs_shape[1],  # W
            )
            out = self.conv(dummy)

        self.output_dim = out.numel()

    def forward(self, x):
        """
        x: (B, H, W, C) -> (B, C, H, W)
        """
        # Ensure correct channel ordering
        if x.ndim == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        
        # Normalize to [0, 1] internally for safety
        x = x / 255.0
        
        return self.conv(x)
        
# --- MLP Policy + Critic (Orthogonal Init Added) ---
class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64): 
        super().__init__()

        # Policy Network
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            # Output layer initialized with small std (0.01) for max entropy start
            layer_init(nn.Linear(hidden_dim, act_dim), std=0.01),
        )

        # Value Network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            # Value output initialized with std 1.0
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def act(self, obs, deterministic=False):
        """
        obs: (B, obs_dim)
        deterministic: If True, pick the argmax action (Greedy).
                       If False, sample from distribution (Stochastic).
        """
        logits = self.actor(obs)
        dist = torch.distributions.Categorical(logits=logits)
        
        if deterministic:
            action = torch.argmax(logits, dim=1) # Take the best action
        else:
            action = dist.sample()               # Roll the dice
            
        logp = dist.log_prob(action)
        value = self.critic(obs).squeeze(-1)
        
        return action, logp, value

    def evaluate(self, obs, actions):
        """
        obs: (B, obs_dim)
        actions: (B,)
        """
        logits = self.actor(obs)
        dist = torch.distributions.Categorical(logits=logits)
        
        logp = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.critic(obs).squeeze(-1)
        return logp, entropy, value

# --- CNN Policy + Critic (Connected to new Extractor) ---
class CNNActorCritic(nn.Module):
    def __init__(self, obs_shape, act_dim, hidden_dim=512):
        super().__init__()

        self.feature_extractor = CNNFeatureExtractor(obs_shape)
        feat_dim = self.feature_extractor.output_dim

        # Actor Head
        self.actor = nn.Sequential(
            layer_init(nn.Linear(feat_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, act_dim), std=0.01),
        )

        # Critic Head
        self.critic = nn.Sequential(
            layer_init(nn.Linear(feat_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def _extract(self, obs):
        return self.feature_extractor(obs)

    def act(self, obs, deterministic=False):
        features = self._extract(obs)
        logits = self.actor(features)

        dist = torch.distributions.Categorical(logits=logits)
        
        if deterministic:
            action = torch.argmax(logits, dim=1) # Take the best action