import numpy as np
import torch
import torch.optim as optim
from copy import deepcopy

from .actor_critic import MLPActorCritic, CNNActorCritic

class FOMAML:
    def __init__(
        self,
        scenario_creator, 
        lr_inner=0.01, # TODO 0.1    
        lr_outer=0.001, # TODO it can be increase not too much 0.001 0.01    
        device="cpu",
        difficulty="medium"
    ):
        self.sc = scenario_creator
        self.difficulty = difficulty
        self.device = torch.device(device)
        self.lr_inner = lr_inner
        
        dummy_env = self.sc.create_env(difficulty, seed=42)
        sample_obs, _ = dummy_env.reset()
        act_dim = dummy_env.action_space.n
        
        if sample_obs.ndim == 1:
            self.use_cnn = False
            obs_dim = int(np.prod(sample_obs.shape))
            self.meta_policy = MLPActorCritic(obs_dim, act_dim).to(self.device)
        else:
            self.use_cnn = True
            obs_shape = sample_obs.shape
            self.meta_policy = CNNActorCritic(obs_shape, act_dim).to(self.device)
            
        self.meta_optimizer = optim.Adam(self.meta_policy.parameters(), lr=lr_outer)
        
        self.fast_policy = deepcopy(self.meta_policy)
        self.fast_policy.to(self.device)
        self.fast_policy.train()

        self.gamma = 0.99
        self.lam = 0.95
        self.vf_coef = 0.5
        self.ent_coef = 0.2 
        
    def _obs_to_tensor(self, state):
        if self.use_cnn:
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
        return state_t 

    def collect_trajectory(self, env, policy, steps=20):
        obs_buf, act_buf, rew_buf, val_buf, logp_buf, done_buf = [], [], [], [], [], []
        
        episode_lens = []       
        episode_rewards = []  
        
        current_len = 0         
        current_ep_rew = 0      

        state, _ = env.reset()
        
        for _ in range(steps):
            state_t = self._obs_to_tensor(state)
            
            with torch.no_grad():
                action, logp, value = policy.act(state_t, deterministic=False)
            
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            current_len += 1
            current_ep_rew += reward

            obs_buf.append(state_t)
            act_buf.append(action)
            rew_buf.append(reward)
            val_buf.append(value)
            logp_buf.append(logp)
            done_buf.append(done)
            
            state = next_state

            if done:
                episode_lens.append(current_len)
                episode_rewards.append(current_ep_rew)
                
                current_len = 0
                current_ep_rew = 0
                state, _ = env.reset()

        last_state_t = self._obs_to_tensor(state)
        with torch.no_grad():
            _, _, last_val = policy.act(last_state_t)
            
        return {
            "obs": torch.cat(obs_buf),
            "act": torch.cat(act_buf),
            "rew": torch.tensor(rew_buf, dtype=torch.float32).to(self.device),
            "val": torch.cat(val_buf),
            "logp": torch.cat(logp_buf),
            "done": torch.tensor(done_buf, dtype=torch.float32).to(self.device),
            "last_val": last_val,
            "ep_lens": episode_lens,      
            "ep_rews": episode_rewards
        }
    # PPO for training
    def compute_loss(self, batch, policy):
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
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = (batch["val"] + adv_t).detach()
        
        new_logp, entropy, new_vals = policy.evaluate(batch["obs"], batch["act"])
        
        pi_loss = -(new_logp * adv_t).mean()
        v_loss = ((new_vals - ret_t) ** 2).mean()
        
        total_loss = pi_loss + self.vf_coef * v_loss - self.ent_coef * entropy.mean()
        return total_loss
    
    # TODO
    #  set the parameters of the models initial ones
    # training
    # take previous fast policy parameters with deepcopy store it for meta update 
    # and then reset the model at the beginning of for loop 
    # import copy

    def meta_train_step(self, task_seeds, k_support=50, k_query=50):
        meta_loss_accum = 0.0

        all_query_lens = []   
        all_query_rews = []

        self.meta_optimizer.zero_grad()
        meta_state_dict = deepcopy(self.meta_policy.state_dict())
        
        for seed in task_seeds:
            env = self.sc.create_env(self.difficulty, seed=seed)
            
            self.fast_policy.load_state_dict(meta_state_dict)
            self.fast_policy.train()

            inner_optim = optim.SGD(self.fast_policy.parameters(), lr=self.lr_inner)            
            
            support_data = self.collect_trajectory(env, self.fast_policy, steps=k_support)
            support_loss = self.compute_loss(support_data, self.fast_policy)
            
            inner_optim.zero_grad()
            support_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.fast_policy.parameters(), max_norm=0.5)
            inner_optim.step()
            
            env.reset(seed=seed) 
            
            query_data = self.collect_trajectory(env, self.fast_policy, steps=k_query)
            
            if len(query_data["ep_lens"]) > 0:
                all_query_lens.extend(query_data["ep_lens"])
                all_query_rews.extend(query_data["ep_rews"])

            query_loss = self.compute_loss(query_data, self.fast_policy)
            
            self.fast_policy.zero_grad()
            query_loss.backward()
            
            for param, meta_param in zip(self.fast_policy.parameters(), self.meta_policy.parameters()):
                if param.grad is not None:
                    if meta_param.grad is None:
                        meta_param.grad = torch.zeros_like(meta_param)
                    meta_param.grad.data.add_(param.grad.data)
                    
            meta_loss_accum += query_loss.item()
            env.close()

        for param in self.meta_policy.parameters():
            if param.grad is not None:
                param.grad.data.div_(len(task_seeds))

        torch.nn.utils.clip_grad_norm_(self.meta_policy.parameters(), max_norm=0.5) 
        self.meta_optimizer.step()
        
        avg_loss = meta_loss_accum / len(task_seeds)

        if len(all_query_rews) > 0:
            avg_rew = np.mean(all_query_rews)  
            avg_steps = np.mean(all_query_lens) 
        else:
            avg_rew = 0.0
            avg_steps = float(k_query) 

        return avg_loss, avg_rew, avg_steps