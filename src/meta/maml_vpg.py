# src/meta/maml_vpg.py

import torch
from copy import deepcopy

from src.actor_critic import MLPActorCritic
from .meta_utils import collect_episodes, compute_policy_loss


class MAMLVPG:
    """
    MAML with REINFORCE (VPG) inner loop.
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

        # Outer optimizer
        self.outer_opt = torch.optim.Adam(
            self.policy.parameters(),
            lr=outer_lr
        )

    # Inner adaptation
    def adapt(self, env, n_episodes: int, max_steps: int = 200):
        """
        Perform one inner-loop gradient step on a single task.
        Returns adapted policy θ'.
        """
        # 1. Collect inner trajectories with current θ
        trajs = collect_episodes(
            env,
            self.policy,
            self.device,
            n_episodes,
            max_steps,
        )

        # 2. Compute inner loss
        inner_loss = compute_policy_loss(
            trajs,
            self.policy,
            self.gamma,
            self.device,
        )

        # 3. Compute gradients - have to keep graph for second order derivation ! 
        grads = torch.autograd.grad(
            inner_loss,
            self.policy.parameters(),
            create_graph=True,
        )

        # 4. Create adapted policy θ'
        adapted_policy = deepcopy(self.policy)

        with torch.no_grad():
            for p, g, p_new in zip(
                self.policy.parameters(),
                grads,
                adapted_policy.parameters()
            ):
                p_new.copy_(p - self.inner_lr * g)

        return adapted_policy

    # Meta-update
    def meta_update(
        self,
        task_envs,
        inner_episodes: int,
        outer_episodes: int,
        max_steps: int = 200,
    ):
        """
        One meta-update over a batch of tasks.
        """
        meta_loss = 0.0

        for env in task_envs:
            # 1. Inner adaptation
            adapted_policy = self.adapt(
                env,
                n_episodes=inner_episodes,
                max_steps=max_steps,
            )

            # 2. Outer rollouts with θ'
            outer_trajs = collect_episodes(
                env,
                adapted_policy,
                self.device,
                outer_episodes,
                max_steps,
            )

            # 3. Outer loss (evaluated on θ')
            loss_outer = compute_policy_loss(
                outer_trajs,
                adapted_policy,
                self.gamma,
                self.device,
            )

            meta_loss += loss_outer

        meta_loss = meta_loss / len(task_envs)

        # 4. Meta optimization
        self.outer_opt.zero_grad()
        meta_loss.backward()
        self.outer_opt.step()

        return meta_loss.item()
