import numpy as np
import torch
import torch.nn as nn

# ---------- Small container for rollouts ----------
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def add(self, state, action, logprob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.state_values.append(value)
        self.rewards.append(reward)
        self.is_terminals.append(done)

    def to_tensors(self, device):
        """Convert stored lists to torch tensors for PPO training."""
        return (
            torch.stack(self.states).to(device),
            torch.tensor(self.actions, dtype=torch.long, device=device),
            torch.tensor(self.logprobs, dtype=torch.float32, device=device),
            torch.tensor(self.rewards, dtype=torch.float32, device=device),
            torch.tensor(self.state_values, dtype=torch.float32, device=device),
            torch.tensor(self.is_terminals, dtype=torch.float32, device=device),
        )
