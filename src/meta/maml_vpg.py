# src/meta/maml_vpg.py

import torch
import torch.nn as nn

from src.actor_critic import MLPActorCritic
from .meta_utils import collect_episodes, compute_policy_loss, fast_adapt

class MAMLVPG:
    """
    MAML algorithm with REINFORCE (VPG) inner loop.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        inner_lr: float,
        outer_lr: float,
        gamma: float,
        device,
    ):
        self.device = device
        self.gamma = gamma
        self.inner_lr = inner_lr

        # Meta-policy θ
        self.policy = MLPActorCritic(obs_dim, act_dim).to(device)

        # Outer loop optimizer
        self.outer_opt = torch.optim.Adam(
            self.policy.parameters(),
            lr=outer_lr
        )

    def adapt(self, env, n_episodes: int, max_steps: int = 200):
        """
        Perform one inner-loop gradient step.
        Returns adapted policy θ' with gradient history connected to θ.
        """
        # 1. Collect support trajectories with current θ
        trajs = collect_episodes(
            env,
            self.policy,
            self.device,
            n_episodes,
            max_steps,
        )

        # 2. Compute inner loss (REINFORCE)
        inner_loss = compute_policy_loss(
            trajs,
            self.policy,
            self.gamma,
            self.device,
        )

        # 3. Compute gradients (create_graph=True is crucial for meta-learning)
        grads = torch.autograd.grad(
            inner_loss,
            self.policy.parameters(),
            create_graph=True,
            allow_unused=True
        )

        # 4. Create adapted policy θ' using graph-preserving helper
        adapted_policy = fast_adapt(self.policy, grads, self.inner_lr)

        return adapted_policy

    def meta_update(
        self,
        task_envs,
        inner_episodes: int,
        outer_episodes: int,
        max_steps: int = 200,
    ):
        """
        Perform one meta-update step over a batch of tasks.
        """
        meta_loss = 0.0
        self.outer_opt.zero_grad()

        for env in task_envs:
            # 1. Inner adaptation (θ -> θ')
            adapted_policy = self.adapt(
                env,
                n_episodes=inner_episodes,
                max_steps=max_steps,
            )

            # 2. Collect query trajectories with θ'
            outer_trajs = collect_episodes(
                env,
                adapted_policy,
                self.device,
                outer_episodes,
                max_steps,
            )

            # 3. Compute outer loss on θ'
            loss_outer = compute_policy_loss(
                outer_trajs,
                adapted_policy,
                self.gamma,
                self.device,
            )

            meta_loss += loss_outer

        # Average loss over the batch
        meta_loss = meta_loss / len(task_envs)

        # 4. Meta-optimization (backprop through the adaptation step)
        meta_loss.backward()
        
        # Clip gradients to improve stability
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        
        self.outer_opt.step()

        return meta_loss.item()