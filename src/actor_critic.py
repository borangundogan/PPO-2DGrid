import numpy as np
import torch
import torch.nn as nn

# ---------- MLP policy + value in one module ----------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int = 4, hidden_dim: int = 64, p_drop: float = 0.1):
        super().__init__()

        # Actor network (policy) # ELU , GELU
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=p_drop),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=p_drop),
            nn.Linear(hidden_dim, act_dim),
            nn.Softmax(dim=-1),
        )

        # Critic (Value function)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def act(self, obs: torch.Tensor):
        """Sample an action from the policy distribution."""
        probs = self.actor(obs)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        logp = dist.log_prob(action)
        value = self.critic(obs)
        return action, logp, value.squeeze(-1)

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        """Compute log-probabilities, entropy, and state values for PPO updates."""
        probs = self.actor(obs)
        dist = torch.distributions.Categorical(probs)
        logp = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(obs)
        return logp, entropy, values.squeeze(-1)