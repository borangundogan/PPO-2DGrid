import torch
import torch.nn as nn
from .utils.utils_rl import layer_init

class CNNFeatureExtractor(nn.Module):
    def __init__(self, channels, height, width):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            self.output_dim = self.network(torch.zeros(1, channels, height, width)).shape[1]

    def forward(self, x):
        return self.network(x / 255.0)

class CNNActorCritic(nn.Module):
    def __init__(self, obs_shape, act_dim, hidden_dim=512):
        super().__init__()
        h, w, c = obs_shape
        
        self.actor_extractor = CNNFeatureExtractor(c, h, w)
        self.critic_extractor = CNNFeatureExtractor(c, h, w)
        
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.actor_extractor.output_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, act_dim), std=0.01),
        )
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.critic_extractor.output_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def _format_obs(self, x):
        if x.ndim == 4 and x.shape[-1] == 3:
            return x.permute(0, 3, 1, 2).float()
        return x.float()

    def act(self, obs, deterministic=False):
        obs = self._format_obs(obs)
        logits = self.actor(self.actor_extractor(obs))
        dist = torch.distributions.Categorical(logits=logits)
        
        action = torch.argmax(logits, dim=1) if deterministic else dist.sample()
        value = self.critic(self.critic_extractor(obs)).squeeze(-1)
        
        return action, dist.log_prob(action), value

    def evaluate(self, obs, actions):
        obs = self._format_obs(obs)
        logits = self.actor(self.actor_extractor(obs))
        dist = torch.distributions.Categorical(logits=logits)
        
        value = self.critic(self.critic_extractor(obs)).squeeze(-1)
        return dist.log_prob(actions), dist.entropy(), value

class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, act_dim), std=0.01),
        )
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def act(self, obs, deterministic=False):
        logits = self.actor(obs)
        dist = torch.distributions.Categorical(logits=logits)
        
        action = torch.argmax(logits, dim=1) if deterministic else dist.sample()
        value = self.critic(obs).squeeze(-1)
        
        return action, dist.log_prob(action), value

    def evaluate(self, obs, actions):
        logits = self.actor(obs)
        dist = torch.distributions.Categorical(logits=logits)
        
        value = self.critic(obs).squeeze(-1)
        return dist.log_prob(actions), dist.entropy(), value