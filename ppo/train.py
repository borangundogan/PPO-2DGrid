import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import time
import argparse
from datetime import datetime

import numpy as np
import torch
import gymnasium as gym
import minigrid

from src.utils.utils import set_seed
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from src.ppo import PPO
from src.utils.utils import get_device
from src.scenario_creator.scenario_creator import ScenarioCreator

from src.metrics.ppo_metrics import compute_episode_stats

# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on MiniGrid environment")

    parser.add_argument("--device", type=str, default="auto", help="cpu or cuda:0")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--update_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--minibatch_size", type=int, default=256)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--total_steps", type=int, default=200_000)
    parser.add_argument("--save_interval", type=int, default=20_000)
    parser.add_argument("--eval_episodes", type=int, default=3)

    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")

    parser.add_argument("--visual_eval", action="store_true",
                        help="Render agent visually after training")

    parser.add_argument("--print_interval", type=int, default=2048,
                        help="Console print frequency (in steps)")

    parser.add_argument(
        "--difficulty",
        type=str,
        default="easy",
        choices=["easy", "medium", "hard", "hardest"],
        help="Environment difficulty level"
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed for training and envs")
    parser.add_argument("--group_timestamp", type=str, default=None, 
                        help="Global timestamp from bash to group experiments")

    return parser.parse_args()


# Evaluation with PPO
def evaluate_policy(agent, env, episodes=3, seed=None):
    rewards = []
    base_seed = seed if seed is not None else 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        obs = np.array(obs, dtype=np.float32)
        done = False
        total_reward = 0

        while not done:
            with torch.no_grad():
                if getattr(agent, "use_cnn", False):
                    obs_t = torch.tensor(obs, dtype=torch.float32,
                                         device=agent.device).unsqueeze(0)
                else:
                    obs_t = torch.tensor(obs, dtype=torch.float32,
                                         device=agent.device).view(1, -1)

                obs_t = obs_t / 255.0  # Normalizasyon
                action, _, _ = agent.ac.act(obs_t, deterministic=True)

            obs, reward, terminated, truncated, _ = env.step(action.item())
            obs = np.array(obs, dtype=np.float32)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)

    return rewards


def visualize_agent(agent, difficulty="easy", episodes=1):
    print(f"[Visualize] Difficulty = {difficulty}")

    sc_gen = ScenarioCreator("src/config/scenario.yaml")
    
    cfg = sc_gen.config["difficulties"][difficulty]
    env_id = cfg["env_id"]
    
    env_kwargs = {**sc_gen.global_cfg, **cfg.get("params", {})}
    env_kwargs["render_mode"] = "human"
    
    env = gym.make(env_id, **env_kwargs)
    
    obs_cfg = sc_gen.get_observation_params()
    
    if obs_cfg.get("fully_observable", False):
        from minigrid.wrappers import FullyObsWrapper
        env = FullyObsWrapper(env)
    else:
        from minigrid.wrappers import RGBImgPartialObsWrapper
        env = RGBImgPartialObsWrapper(env)
        
    from minigrid.wrappers import ImgObsWrapper
    env = ImgObsWrapper(env)
    
    if obs_cfg.get("flatten", False):
        from gymnasium.wrappers import FlattenObservation
        env = FlattenObservation(env)
        
    from src.wrappers.three_action_wrapper import ThreeActionWrapper
    env = ThreeActionWrapper(env)

    device = agent.device
    actor = agent.ac

    for ep in range(episodes):
        obs, _ = env.reset()
        obs = np.array(obs, dtype=np.float32)
        total_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                if getattr(agent, "use_cnn", False):
                    obs_t = torch.tensor(obs, dtype=torch.float32,
                                         device=device).unsqueeze(0)
                else:
                    obs_t = torch.tensor(obs, dtype=torch.float32,
                                         device=device).view(1, -1)

                obs_t = obs_t / 255.0
                action, _, _ = actor.act(obs_t, deterministic=True)

            obs, reward, terminated, truncated, _ = env.step(action.item())
            obs = np.array(obs, dtype=np.float32)
            total_reward += reward
            done = terminated or truncated
            time.sleep(0.08)

        print(f"[Visual Eval] Episode {ep+1}: Reward = {total_reward:.3f}")

    env.close()


# Training Loop
def train_minigrid(args):
    set_seed(args.seed)
    print(f"[Seed] Using seed = {args.seed}")
    
    print("============================================================================================")
    device = get_device(args.device)

    sc_gen = ScenarioCreator("src/config/scenario.yaml")
    env = sc_gen.create_env(difficulty=args.difficulty, seed=args.seed)
    print(f"Loaded environment from ScenarioCreator | Difficulty: {args.difficulty}")

    sample_obs, _ = env.reset(seed=args.seed)
    sample_obs = np.array(sample_obs, dtype=np.float32)
    print(f"Sample obs shape: {sample_obs.shape}, min={sample_obs.min()}, max={sample_obs.max()}")

    agent = PPO(
        env=env,
        lr=args.lr,
        gamma=args.gamma,
        lam=args.lam,
        clip_eps=args.clip_eps,
        update_epochs=args.update_epochs,
        batch_size=args.batch_size,
        minibatch_size=args.minibatch_size,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        device=device,
    )
    
    env_id = sc_gen.config["difficulties"][args.difficulty]["env_id"]

    if args.group_timestamp:
        timestamp = args.group_timestamp
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment_name = f"{env_id}_{args.difficulty}_{timestamp}"

    # checkpoints/ExperimentName/seed_123/
    ckpt_subdir = os.path.join(args.ckpt_dir, experiment_name, f"seed_{args.seed}")
    
    # tb_logs/ExperimentName/seed_123/
    tb_dir = os.path.join("tb_logs", experiment_name, f"seed_{args.seed}")

    os.makedirs(ckpt_subdir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    writer_tb = SummaryWriter(log_dir=tb_dir)
    ckpt_path = os.path.join(ckpt_subdir, "ppo_model.pth")

    step_count = 0
    start_time = time.time()

    print(f"Training started on {device} for {args.total_steps:,} total steps")
    print(f"Experiment Group: {experiment_name}")
    print(f"Saving to: {ckpt_subdir}")
    print("============================================================================================")

    eval_env = sc_gen.create_env(args.difficulty, seed=args.seed + 999)

    # Initialize best reward tracking
    best_reward = -float('inf')

    while step_count < args.total_steps:

        last_value = agent.collect_rollouts()

        update_stats = agent.update(last_value)
        
        step_count += agent.batch_size

        eval_rewards = evaluate_policy(
            agent,
            eval_env,
            episodes=args.eval_episodes,
            seed=args.seed + 999
        )
        avg_r = np.mean(eval_rewards)

        # Save Best Model Logic
        if avg_r > best_reward:
            best_reward = avg_r
            best_model_path = os.path.join(ckpt_subdir, "best_model.pth")
            torch.save(agent.ac.state_dict(), best_model_path)
            print(f"[*] New best model saved! Reward: {best_reward:.3f} -> {best_model_path}")

        elapsed_min = (time.time() - start_time) / 60

        writer_tb.add_scalar("reward/avg_eval_reward", avg_r, step_count)

        writer_tb.add_scalar("loss/policy_loss", update_stats["pi_loss"], step_count)
        writer_tb.add_scalar("loss/value_loss", update_stats["v_loss"], step_count)
        writer_tb.add_scalar("loss/entropy", update_stats["entropy"], step_count)

        writer_tb.add_scalar("diagnostics/kl", update_stats["kl"], step_count)
        writer_tb.add_scalar("diagnostics/clipfrac", update_stats["clipfrac"], step_count)
        writer_tb.add_scalar("diagnostics/gradnorm", update_stats["gradnorm"], step_count)

        stats = compute_episode_stats(
            agent.episode_returns[-10:],
            agent.episode_lengths[-10:]
        )

        writer_tb.add_scalar("stats/episode_return_mean", stats["episode_return_mean"], step_count)
        writer_tb.add_scalar("stats/episode_length_mean", stats["episode_length_mean"], step_count)
        
        if len(agent.episode_returns) >= 10:
            writer_tb.add_histogram("hist/episode_rewards",
                                    np.array(agent.episode_returns[-50:]),
                                    step_count)
            writer_tb.add_histogram("hist/episode_lengths",
                                    np.array(agent.episode_lengths[-50:]),
                                    step_count)

            fig = plt.figure()
            plt.scatter(agent.episode_lengths[-50:], agent.episode_returns[-50:], c="green")
            plt.xlabel("Steps")
            plt.ylabel("Reward")
            plt.title("Reward vs Episode Length")
            writer_tb.add_figure("fig/reward_vs_steps", fig, step_count)
            plt.close(fig)
        
        # Log frequency check
        if step_count % args.print_interval == 0 or step_count >= args.total_steps:
            total_loss = update_stats["pi_loss"] + update_stats["v_loss"]

            print(
                f"[Steps: {step_count:>7}] | "
                f"Reward: {avg_r:.3f} | "
                f"Loss: {total_loss:.4f} | "
                f"π: {update_stats['pi_loss']:.4f} | "
                f"V: {update_stats['v_loss']:.4f} | "
                f"Ent: {update_stats['entropy']:.4f} | "
                f"KL: {update_stats['kl']:.6f} | "
                f"Clip: {update_stats['clipfrac']:.3f} | "
                f"GradNorm: {update_stats['gradnorm']:.3f} | "
                f"Time: {elapsed_min:.2f}m"
            )

        if step_count % args.save_interval == 0 or step_count >= args.total_steps:
            torch.save(agent.ac.state_dict(), ckpt_path)
            print(f"Model checkpoint saved at step {step_count} -> {ckpt_path}")

    # Visual eval if requested
    if args.visual_eval:
        visualize_agent(agent, difficulty=args.difficulty)

if __name__ == "__main__":
    args = parse_args()
    train_minigrid(args)