import torch
import torch.nn as nn
import torch.nn.functional as F


# CNN Feature Extractor (Dynamic output dim)
class CNNFeatureExtractor(nn.Module):
    def __init__(self, obs_shape, input_channels=3):
        """
        obs_shape: (H, W, C) coming directly from env reset().
        """
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # DYNAMIC FEATURE SIZE COMPUTATION
        # Create dummy input: (1, C, H, W)
        dummy = torch.zeros(
            1,
            obs_shape[2],  # C
            obs_shape[0],  # H
            obs_shape[1],  # W
        )

        with torch.no_grad():
            out = self.conv(dummy)

        # Compute flattened size
        self.output_dim = out.numel()

    def forward(self, x):
        """
        x: (B, H, W, C)
        convert → (B, C, H, W)
        """
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        return x.reshape(x.size(0), -1)  # flatten
        

# MLP Policy + Critic (Flattened input)
class MLPActorCritic(nn.Module):
    # UPDATED: Increased default hidden_dim to 256 for better capacity on Hard tasks
    def __init__(self, obs_dim, act_dim, hidden_dim=256): 
        super().__init__()

        # Policy
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(), # Tanh is often preferred for PPO value stability
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
        )

        # Value
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    # PPO-compatible API
    def act(self, obs):
        """
        obs: (B, obs_dim)
        """
        logits = self.actor(obs)
        dist = torch.distributions.Categorical(logits=logits)
        
        action = dist.sample()
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


# CNN Policy + Critic (Image input)
class CNNActorCritic(nn.Module):
    def __init__(self, obs_shape, act_dim, hidden_dim=256):
        """
        obs_shape: (H, W, C)
        """
        super().__init__()

        self.feature_extractor = CNNFeatureExtractor(obs_shape)
        feat_dim = self.feature_extractor.output_dim

        # Policy network
        self.actor = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )

        # Value function
        self.critic = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    # helpers
    def _extract(self, obs):
        # obs: (B, H, W, C)
        return self.feature_extractor(obs)

    # PPO API
    def act(self, obs):
        features = self._extract(obs)
        logits = self.actor(features)

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)

        value = self.critic(features).squeeze(-1)
        return action, logp, value

    def evaluate(self, obs, actions):
        features = self._extract(obs)
        logits = self.actor(features)

        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(actions)
        entropy = dist.entropy()

        value = self.critic(features).squeeze(-1)
        return logp, entropy, value