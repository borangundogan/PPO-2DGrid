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
        dummy = torch.zeros(
            1,
            obs_shape[2],  # C
            obs_shape[0],  # H
            obs_shape[1],  # W
        )

        with torch.no_grad():
            out = self.conv(dummy)

        self.output_dim = out.numel()

    def forward(self, x):
        """
        x: (B, H, W, C) -> (B, C, H, W)
        """
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        return x.reshape(x.size(0), -1)  # flatten
        

# MLP Policy + Critic (Flattened input)
class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256): 
        super().__init__()

        # Policy
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
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
        
        # Note: In deterministic mode, entropy is technically 0, but we return the sampled entropy or None. Usually ignored during eval.
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
        super().__init__()

        self.feature_extractor = CNNFeatureExtractor(obs_shape)
        feat_dim = self.feature_extractor.output_dim

        self.actor = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _extract(self, obs):
        return self.feature_extractor(obs)

    def act(self, obs, deterministic=False):
        features = self._extract(obs)
        logits = self.actor(features)

        dist = torch.distributions.Categorical(logits=logits)
        
        if deterministic:
            action = torch.argmax(logits, dim=1) # Take the best action
        else:
            action = dist.sample()               # Roll the dice
            
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