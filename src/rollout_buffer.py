import torch

class RolloutBuffer:
    def __init__(self, buffer_size, obs_shape, device, is_discrete=True):
        self.states = torch.zeros((buffer_size, *obs_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros(buffer_size, dtype=torch.long if is_discrete else torch.float32, device=device)
        self.logprobs = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.values = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.max_size = buffer_size
        self.ptr = 0

    def add(self, state, action, logprob, value, reward, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.logprobs[self.ptr] = logprob
        self.values[self.ptr] = value
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size

    def get(self):
        self.ptr = 0
        return (
            self.states,
            self.actions,
            self.logprobs,
            self.rewards,
            self.values,
            self.dones,
        )