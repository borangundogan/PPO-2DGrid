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

        # Inspect one observation to decide MLP (flattened) vs CNN (image)
        sample_obs, _ = env.reset()
        act_dim = env.action_space.n

        if sample_obs.ndim == 1:
            # Flattened vector input → use MLP policy
            self.use_cnn = False
            obs_dim = int(np.prod(sample_obs.shape))
            self.ac = MLPActorCritic(obs_dim, act_dim).to(self.device)
            print(f"[PPO] Using MLPActorCritic, obs_dim={obs_dim}")
        else:
            # Image-like input (H, W, C) → use CNN policy
            self.use_cnn = True
            obs_shape = sample_obs.shape  # (H, W, C)
            self.ac = CNNActorCritic(obs_shape, act_dim).to(self.device)
            print(f"[PPO] Using CNNActorCritic, obs_shape={obs_shape}")

        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

        # For logging
        self.episode_returns = []
        self.episode_lengths = []

    def _obs_to_tensor(self, state):
        """Convert raw env state to torch tensor, handling MLP vs CNN."""
        if self.use_cnn:
            # state: (H, W, C) → (1, H, W, C)
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            # state: (obs_dim,) → (1, obs_dim)
            state_t = (
                torch.tensor(state, dtype=torch.float32, device=self.device)
                .view(1, -1)
            )
        # MiniGrid obs are uint8 images; normalize to [0,1]
        state_t = state_t / 255.0
        return state_t

    def collect_rollouts(self):
        self.buffer.clear()
        state, _ = self.env.reset()

        total_steps = 0
        ep_return = 0
        ep_length = 0

        while total_steps < self.batch_size:
            state_t = self._obs_to_tensor(state)

            with torch.no_grad():
                action, logp, value = self.ac.act(state_t)

            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated

            # Store state without batch dimension
            self.buffer.add(state_t.squeeze(0), action, logp, value, reward, done)

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

        # This ensures images pass through the CNN extractor first.
        last_state_t = self._obs_to_tensor(state)
        with torch.no_grad():
            _, _, last_val_tensor = self.ac.act(last_state_t)
            last_value = last_val_tensor.item()
        
        return last_value

    def compute_gae(self, rewards, values, dones, last_value):
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
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
        (
            states,
            actions,
            logprobs_old,
            rewards,
            values_old,
            dones,
        ) = self.buffer.to_tensors(self.device)

        adv, returns = self.compute_gae(
            rewards.cpu().numpy(),
            values_old.cpu().numpy(),
            dones.cpu().numpy(),
            last_value,
        )

        adv = torch.tensor(adv, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        N = states.shape[0]
        idxs = np.arange(N)

        total_pi = total_v = total_ent = 0
        total_kl = total_clip = total_gnorm = 0
        nbatches = 0

        for _ in range(self.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, N, self.minibatch_size):
                mb_idx = idxs[start:start+self.minibatch_size]
                mb = lambda x: x[mb_idx]

                logp_new, entropy, values = self.ac.evaluate(
                    mb(states), mb(actions)
                )

                ratio = torch.exp(logp_new - mb(logprobs_old))
                surr1 = ratio * mb(adv)
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb(adv)

                pi_loss = -torch.min(surr1, surr2).mean()
                v_loss = ((values - mb(returns))**2).mean()
                loss = pi_loss + self.vf_coef*v_loss - self.ent_coef*entropy.mean()

                approx_kl = (mb(logprobs_old) - logp_new).mean()
                clipfrac = (torch.abs(ratio - 1.0) > self.clip_eps).float().mean()

                self.optimizer.zero_grad()
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
            total_pi,
            total_v,
            total_ent,
            total_kl,
            total_clip,
            total_gnorm,
            nbatches
        )

    def train(self, total_steps=100_000):
        steps_done = 0
        while steps_done < total_steps:
            last_value = self.collect_rollouts()
            self.update(last_value)
            steps_done += self.batch_size
            print(f"Steps: {steps_done} done.")