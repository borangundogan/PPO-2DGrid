import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .actor_critic import MLPActorCritic, CNNActorCritic
from .rollout_buffer import RolloutBuffer
from src.metrics.ppo_metrics import aggregate_ppo_update_metrics

class PPO:
    def __init__(
        self,
        env,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        update_epochs=10,
        batch_size=2048,
        minibatch_size=256,
        vf_coef=0.5,
        ent_coef=0.01,
        device="cpu",
    ):
        self.env = env
        self.device = torch.device(device)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

        sample_obs, _ = env.reset()
        act_dim = env.action_space.n

        if sample_obs.ndim == 1:
            self.use_cnn = False
            self.obs_shape = (int(np.prod(sample_obs.shape)),)
            self.ac = MLPActorCritic(self.obs_shape[0], act_dim).to(self.device)
        else:
            self.use_cnn = True
            self.obs_shape = sample_obs.shape
            self.ac = CNNActorCritic(self.obs_shape, act_dim).to(self.device)

        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)
        self.buffer = RolloutBuffer(
            buffer_size=self.batch_size,
            obs_shape=self.obs_shape,
            device=self.device,
            is_discrete=True
        )

        self.episode_returns = []
        self.episode_lengths = []

    def _obs_to_tensor(self, state):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        if self.use_cnn:
            return state_t.unsqueeze(0)
        return state_t.view(1, -1)

    def collect_rollouts(self):
        state, _ = self.env.reset()
        total_steps = 0
        ep_return = 0
        ep_length = 0

        while total_steps < self.batch_size:
            state_t = self._obs_to_tensor(state)

            with torch.no_grad():
                action, logp, value = self.ac.act(state_t, deterministic=False)

            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated

            self.buffer.add(
                state_t.squeeze(0),
                action.squeeze(),
                logp.squeeze(),
                value.squeeze(),
                torch.tensor(reward, dtype=torch.float32, device=self.device),
                torch.tensor(done, dtype=torch.float32, device=self.device)
            )

            ep_return += reward
            ep_length += 1
            state = next_state
            total_steps += 1

            if done:
                self.episode_returns.append(ep_return)
                self.episode_lengths.append(ep_length)
                state, _ = self.env.reset()
                ep_return = 0
                ep_length = 0

        last_state_t = self._obs_to_tensor(state)
        with torch.no_grad():
            _, _, last_val_tensor = self.ac.act(last_state_t)
            last_value = last_val_tensor.item()
        
        return last_value

    def compute_gae(self, rewards, values, dones, last_value):
        T = rewards.size(0)
        adv = torch.zeros_like(rewards)
        gae = 0.0

        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            next_val = last_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_val * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            adv[t] = gae
            
        returns = values + adv
        return adv, returns

    def update(self, last_value):
        states, actions, logprobs_old, rewards, values_old, dones = self.buffer.get()
        adv, returns = self.compute_gae(rewards, values_old, dones, last_value)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        N = states.shape[0]

        total_pi = total_v = total_ent = total_kl = total_clip = total_gnorm = 0.0
        nbatches = 0

        for _ in range(self.update_epochs):
            idxs = torch.randperm(N, device=self.device)
            for start in range(0, N, self.minibatch_size):
                mb_idx = idxs[start : start + self.minibatch_size]
                mb = lambda x: x[mb_idx]

                logp_new, entropy, values = self.ac.evaluate(mb(states), mb(actions))
                values = values.squeeze(-1)

                ratio = torch.exp(logp_new - mb(logprobs_old))
                surr1 = ratio * mb(adv)
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb(adv)

                pi_loss = -torch.min(surr1, surr2).mean()
                v_loss = ((values - mb(returns))**2).mean()
                loss = pi_loss + self.vf_coef * v_loss - self.ent_coef * entropy.mean()

                with torch.no_grad():
                    approx_kl = (mb(logprobs_old) - logp_new).mean()
                    clipfrac = (torch.abs(ratio - 1.0) > self.clip_eps).float().mean()

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                self.optimizer.step()

                total_pi += pi_loss.item()
                total_v += v_loss.item()
                total_ent += entropy.mean().item()
                total_kl += approx_kl.item()
                total_clip += clipfrac.item()
                total_gnorm += grad_norm.item()
                nbatches += 1

        return aggregate_ppo_update_metrics(
            total_pi, total_v, total_ent, total_kl, total_clip, total_gnorm, nbatches
        )

    def train(self, total_steps=100_000):
        steps_done = 0
        while steps_done < total_steps:
            last_value = self.collect_rollouts()
            self.update(last_value)
            steps_done += self.batch_size