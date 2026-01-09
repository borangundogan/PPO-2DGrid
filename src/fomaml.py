import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

from .actor_critic import MLPActorCritic, CNNActorCritic


class FOMAML:
    def __init__(
        self,
        scenario_creator, # Need this to spawn tasks on the fly
        lr_inner=0.01,    # Alpha: Step size for inner adaptation
        lr_outer=3e-4,    # Beta: Step size for meta-update
        device="cpu",
        difficulty="medium"
    ):
        self.sc = scenario_creator
        self.difficulty = difficulty
        self.device = torch.device(device)
        self.lr_inner = lr_inner
        
        # Initialize the Main Meta-Policy (Theta)
        # We create a dummy env just to get shapes
        dummy_env = self.sc.create_env(difficulty, seed=42)
        sample_obs, _ = dummy_env.reset()
        act_dim = dummy_env.action_space.n
        
        # Auto-detect Architecture (Same as PPO)
        if sample_obs.ndim == 1:
            self.use_cnn = False
            obs_dim = int(np.prod(sample_obs.shape))
            self.meta_policy = MLPActorCritic(obs_dim, act_dim).to(self.device)
        else:
            self.use_cnn = True
            obs_shape = sample_obs.shape
            self.meta_policy = CNNActorCritic(obs_shape, act_dim).to(self.device)
            
        # Outer Optimizer (Adam)
        self.meta_optimizer = optim.Adam(self.meta_policy.parameters(), lr=lr_outer)
        
        # Parameters
        self.gamma = 0.99
        self.lam = 0.95
        self.vf_coef = 0.5
        self.ent_coef = 0.01
        self.clip_eps = 0.2
        
    def _obs_to_tensor(self, state):
        """Helper to convert numpy obs to tensor"""
        if self.use_cnn:
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
        return state_t / 255.0

    def collect_trajectory(self, env, policy, steps=20):
        """
        Collects K steps for either Support or Query set.
        Returns data in a simple dict or buffer.
        """
        obs_buf, act_buf, rew_buf, val_buf, logp_buf, done_buf = [], [], [], [], [], []
        
        state, _ = env.reset() # Note: Env is already seeded before calling this
        
        for _ in range(steps):
            state_t = self._obs_to_tensor(state)
            
            with torch.no_grad():
                # Important: Exploration is ON (deterministic=False) during meta-training
                action, logp, value = policy.act(state_t, deterministic=False)
            
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            obs_buf.append(state_t)
            act_buf.append(action)
            rew_buf.append(reward)
            val_buf.append(value)
            logp_buf.append(logp)
            done_buf.append(done)
            
            state = next_state
            if done:
                state, _ = env.reset()

        # Compute Last Value for GAE
        last_state_t = self._obs_to_tensor(state)
        with torch.no_grad():
            _, _, last_val = policy.act(last_state_t)
            
        # Stack tensors
        return {
            "obs": torch.cat(obs_buf),
            "act": torch.cat(act_buf),
            "rew": torch.tensor(rew_buf, dtype=torch.float32).to(self.device),
            "val": torch.cat(val_buf),
            "logp": torch.cat(logp_buf),
            "done": torch.tensor(done_buf, dtype=torch.float32).to(self.device),
            "last_val": last_val
        }

    def compute_loss(self, batch, policy):
        """
        Computes Standard PPO Loss on a batch of data.
        """
        # 1. GAE Calculation
        rews = batch["rew"].cpu().numpy()
        vals = batch["val"].cpu().numpy()
        dones = batch["done"].cpu().numpy()
        last_val = batch["last_val"].item()
        
        adv = np.zeros_like(rews)
        gae = 0.0
        for t in reversed(range(len(rews))):
            mask = 1.0 - dones[t]
            next_v = last_val if t == len(rews) - 1 else vals[t + 1]
            delta = rews[t] + self.gamma * next_v * mask - vals[t]
            gae = delta + self.gamma * self.lam * mask * gae
            adv[t] = gae
            
        adv_t = torch.tensor(adv, dtype=torch.float32, device=self.device)
        ret_t = batch["val"] + adv_t # Returns = Value + Advantage
        
        # 2. Re-evaluate actions to get new log_probs and entropy (Gradient flow happens here)
        # Note: We pass the observations back into the network!
        new_logp, entropy, new_vals = policy.evaluate(batch["obs"], batch["act"])
        
        # 3. PPO Loss
        ratio = torch.exp(new_logp - batch["logp"])
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t
        
        pi_loss = -torch.min(surr1, surr2).mean()
        v_loss = ((new_vals - ret_t) ** 2).mean()
        
        total_loss = pi_loss + self.vf_coef * v_loss - self.ent_coef * entropy.mean()
        return total_loss

    def meta_train_step(self, task_seeds, k_support=20, k_query=20):
        """
        Executes one Meta-Training Step (Outer Loop)
        1. For each task:
           - Adapt (Inner Loop) -> Get Theta'
           - Validate (Query) -> Get Loss(Theta')
        2. Average Query Losses
        3. Update Theta
        """
        meta_loss_accum = 0.0
        self.meta_optimizer.zero_grad()
        
        # Iterate over tasks (e.g., 4 tasks in a batch)
        for seed in task_seeds:
            # --- STEP 1: Support Set (Inner Loop) ---
            # Create fresh env for this specific task
            env = self.sc.create_env(self.difficulty, seed=seed)
            
            # Create a COPY of the model (Theta')
            # This is efficient enough for small models and essential for FOMAML
            fast_policy = deepcopy(self.meta_policy)
            fast_policy.train()
            
            # Inner Optimizer (SGD is standard for Inner Loop)
            inner_optim = optim.SGD(fast_policy.parameters(), lr=self.lr_inner)
            
            # Collect Support Data (Theta)
            support_data = self.collect_trajectory(env, self.meta_policy, steps=k_support)
            
            # Compute Support Loss
            # IMPORTANT: We evaluate the support data on the FAST policy to get gradients
            # to update the FAST policy.
            support_loss = self.compute_loss(support_data, fast_policy)
            
            # Update Theta -> Theta'
            inner_optim.zero_grad()
            support_loss.backward()
            inner_optim.step()
            
            # --- STEP 2: Query Set (Outer Evaluation) ---
            # Reset env to SAME seed to test adaptation
            env.reset(seed=seed) 
            
            # Collect Query Data using Theta' (Adapted Policy)
            query_data = self.collect_trajectory(env, fast_policy, steps=k_query)
            
            # Compute Query Loss
            # We must verify if "fast_policy" is detached or not. 
            # For FOMAML, we use the gradients of this loss to update the ORIGINAL model.
            # In PyTorch, we can do this by running the query data through the ORIGINAL model
            # but treating the "labels" (advantages) as coming from the adapted trajectory.
            
            # SIMPLIFIED FOMAML Implementation:
            # We calculate gradients on fast_policy, then transfer them to meta_policy.
            query_loss = self.compute_loss(query_data, fast_policy)
            query_loss.backward()
            
            # Transfer Gradients: Theta'.grad -> Theta.grad
            for param, meta_param in zip(fast_policy.parameters(), self.meta_policy.parameters()):
                if meta_param.grad is None:
                    meta_param.grad = torch.zeros_like(meta_param)
                if param.grad is not None:
                    # Add query gradients to meta gradients
                    meta_param.grad.data.add_(param.grad.data)
                    
            meta_loss_accum += query_loss.item()
            env.close()

        # --- STEP 3: Meta-Update ---
        # Normalize gradients by batch size
        for param in self.meta_policy.parameters():
            if param.grad is not None:
                param.grad.data.div_(len(task_seeds))
                
        self.meta_optimizer.step()
        
        return meta_loss_accum / len(task_seeds)