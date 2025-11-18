import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .actor_critic import ActorCritic
from .rollout_buffer import RolloutBuffer


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

        sample_obs, _ = env.reset()
        obs_dim = int(np.prod(sample_obs.shape))
        act_dim = env.action_space.n

        self.ac = ActorCritic(obs_dim, act_dim).to(self.device)
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

        # --- CRITICAL for TensorBoard ---
        self.episode_returns = []
        self.episode_lengths = []


    def collect_rollouts(self):
        self.buffer.clear()
        state, _ = self.env.reset()

        total_steps = 0
        ep_return = 0
        ep_length = 0

        while total_steps < self.batch_size:

            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
            state_t = state_t / 255.0

            with torch.no_grad():
                action, logp, value = self.ac.act(state_t)

            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated

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


    def compute_gae(self, rewards, values, dones, last_value):
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * last_value * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            adv[t] = gae
            last_value = values[t]

        returns = values + adv
        return adv, returns


    def update(self):

        (
            states,
            actions,
            logprobs_old,
            rewards,
            values_old,
            dones,
        ) = self.buffer.to_tensors(self.device)

        with torch.no_grad():
            last_value = self.ac.critic(states[-1].unsqueeze(0)).item()

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

        total_pi_loss = 0
        total_v_loss = 0
        total_entropy = 0
        n_batches = 0

        for _ in range(self.update_epochs):
            np.random.shuffle(idxs)

            for start in range(0, N, self.minibatch_size):

                mb_idx = idxs[start:start+self.minibatch_size]
                mb = lambda x: x[mb_idx]

                logp_new, entropy, values = self.ac.evaluate(mb(states), mb(actions))

                ratio = torch.exp(logp_new - mb(logprobs_old))

                surr1 = ratio * mb(adv)
                surr2 = torch.clamp(ratio, 1.0-self.clip_eps, 1.0+self.clip_eps) * mb(adv)

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((values - mb(returns)) ** 2).mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                self.optimizer.step()

                total_pi_loss += policy_loss.item()
                total_v_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_batches += 1

        avg_pi = total_pi_loss / n_batches
        avg_v = total_v_loss / n_batches
        avg_ent = total_entropy / n_batches

        print(f"   PPO update | π_loss: {avg_pi:.4f} | V_loss: {avg_v:.4f} | Entropy: {avg_ent:.4f}")

        return avg_pi, avg_v, avg_ent


    def train(self, total_steps=100_000):
        steps_done = 0
        while steps_done < total_steps:
            self.collect_rollouts()
            self.update()
            steps_done += self.batch_size
            print(f"Steps: {steps_done} done.")
