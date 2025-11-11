import os
import csv
import time
import argparse
from datetime import datetime

import numpy as np
import torch
import gymnasium as gym
import minigrid
from gymnasium.wrappers import FlattenObservation
from minigrid.wrappers import FullyObsWrapper, RGBImgPartialObsWrapper, ImgObsWrapper
import matplotlib.pyplot as plt

from src.ppo import PPO
from src.utils import get_device

from src.scenario_creator.scenario_creator import ScenarioCreator


# ----------------------------------------------------
# CLI arguments
# ----------------------------------------------------
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
    parser.add_argument("--log_interval", type=int, default=5_000)
    parser.add_argument("--save_interval", type=int, default=20_000)
    parser.add_argument("--eval_episodes", type=int, default=3)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--visual_eval", action="store_true", help="Render agent visually after training")
    parser.add_argument("--print_interval", type=int, default=1000, help="Console print frequency (in steps)")
    parser.add_argument("--difficulty", type=str, default="easy", help="Environment difficulty level (easy, medium, hard)")
    return parser.parse_args()


# ----------------------------------------------------
# Evaluation functions
# ----------------------------------------------------
def evaluate_policy(agent, env, episodes=3):
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=agent.device).view(1, -1)
                obs_t = obs_t / 255.0 # # normalize the input observations !
                action, _, _ = agent.ac.act(obs_t)
            obs, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    return rewards


def visualize_agent(agent, env_name="MiniGrid-Empty-8x8-v0", model_path=None, episodes=1):
    print("\nStarting visual evaluation")
    env = gym.make(env_name, render_mode="human")
    env = FullyObsWrapper(env)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = env.action_space.n

    if model_path:
        from src.actor_critic import ActorCritic
        actor = ActorCritic(obs_dim, act_dim)
        actor.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        actor.eval()
    else:
        actor = agent.ac

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).view(1, -1)
                action, _, _ = actor.act(obs_t)
            obs, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            done = terminated or truncated
            time.sleep(0.1)
        print(f"Episode {ep+1}: total_reward = {total_reward:.3f}")
    env.close()


# ----------------------------------------------------
# Main training loop
# ----------------------------------------------------
def train_minigrid(args):
    print("============================================================================================")
    device = get_device()

    # Create environment
    sc_gen = ScenarioCreator("src/config/scenario.yaml")
    env = sc_gen.create_env(difficulty=args.difficulty)
    print(f"Loaded environment from ScenarioCreator | Difficulty: {args.difficulty}")
    # env = FlattenObservation(env)  

    sample_obs, _ = env.reset()
    obs_dim = int(np.prod(sample_obs.shape))
    print(f"Observation shape: {obs_dim}")
    print(f"Sample obs shape: {obs_dim}, min={sample_obs.min()}, max={sample_obs.max()}")

    # PPO agent
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

    # Unique timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_id = sc_gen.config["difficulties"][args.difficulty]["env_id"]
    run_id = f"{env_id}_{args.difficulty}_{timestamp}"


    # Create subdirectories for this run
    log_subdir = os.path.join(args.log_dir, run_id)
    ckpt_subdir = os.path.join(args.ckpt_dir, run_id)
    os.makedirs(log_subdir, exist_ok=True)
    os.makedirs(ckpt_subdir, exist_ok=True)

    # File paths
    log_path = os.path.join(log_subdir, "training.csv")
    ckpt_path = os.path.join(ckpt_subdir, "ppo_model.pth")

    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["step", "avg_reward", "time_min"])

    step_count = 0
    episode_rewards = []
    start_time = time.time()

    print(f"Training started on {device} for {args.total_steps:,} total steps")
    print(f"Environment: {env}")
    print(f"Run name: {run_id}")
    print("============================================================================================")


    while step_count < args.total_steps:
        print("Collecting rollouts...")
        agent.collect_rollouts()
        print("Updating PPO...")
        agent.update()
        step_count += agent.batch_size

        # Evaluate and log
        eval_rewards = evaluate_policy(agent, env, episodes=args.eval_episodes)
        avg_r = np.mean(eval_rewards)
        episode_rewards.append(avg_r)

        elapsed_min = (time.time() - start_time) / 60
        writer.writerow([step_count, round(avg_r, 3), round(elapsed_min, 2)])
        log_file.flush()

        if step_count % args.print_interval == 0 or step_count >= args.total_steps:
            print(f"[Steps: {step_count:>7}] | Avg reward: {avg_r:.3f} | Time: {elapsed_min:.2f} min")

        if step_count % args.save_interval == 0 or step_count >= args.total_steps:
            torch.save(agent.ac.state_dict(), ckpt_path)
            print(f"Model checkpoint saved at step {step_count} -> {ckpt_path}")

    # Cleanup
    log_file.close()
    env.close()

    total_time = (time.time() - start_time) / 60
    print("============================================================================================")
    print(f"Training finished ✅ | Total time: {total_time:.2f} min")
    print(f"Logs saved to: {log_path}")
    print(f"Model saved to: {ckpt_path}")
    print("============================================================================================")

    # --------------------------
    # Plot training progress
    # --------------------------
    try:
        import pandas as pd
        df = pd.read_csv(log_path)
        plt.figure(figsize=(8, 4))
        plt.plot(df["step"], df["avg_reward"], label="Average Reward", color="green")
        plt.xlabel("Training Steps")
        plt.ylabel("Average Reward")
        plt.title(f"PPO Training Progress - {args.env_name}")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(log_subdir, "reward_plot.png")
        plt.savefig(plot_path)
        print(f"Plot saved to: {plot_path}")
        plt.show()
    except Exception as e:
        print(f"Plotting skipped due to error: {e}")

    if args.visual_eval:
        visualize_agent(agent, env_name=args.env_name, model_path=ckpt_path, episodes=2)


# ----------------------------------------------------
# Entry point
# ----------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    train_minigrid(args)
