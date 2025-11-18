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
from torch.utils.tensorboard import SummaryWriter

from src.ppo import PPO
from src.utils import get_device
from src.scenario_creator.scenario_creator import ScenarioCreator


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


def evaluate_policy(agent, env, episodes=3):
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=agent.device).view(1, -1)
                obs_t = obs_t / 255.0
                action, _, _ = agent.ac.act(obs_t)
            obs, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    return rewards


def visualize_agent(agent, difficulty="easy", model_path=None, episodes=1):
    sc_gen = ScenarioCreator("src/config/scenario.yaml")
    cfg = sc_gen.config["difficulties"][difficulty]
    params = cfg.get("params", {}).copy()
    params["render_mode"] = "human"
    env = gym.make(cfg["env_id"], **params)

    obs_cfg = sc_gen.config.get("observation", {})
    if obs_cfg.get("fully_observable", False):
        env = FullyObsWrapper(env)
    else:
        env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    if obs_cfg.get("flatten", False):
        env = FlattenObservation(env)

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = env.action_space.n

    device = agent.device if hasattr(agent, "device") else torch.device("cpu")
    if model_path:
        from src.actor_critic import ActorCritic
        actor = ActorCritic(obs_dim, act_dim).to(device)
        actor.load_state_dict(torch.load(model_path, map_location=device))
        actor.eval()
    else:
        actor = agent.ac

    for ep in range(episodes):
        obs, _ = env.reset()
        done, total_reward = False, 0.0
        while not done:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).view(1, -1)
                obs_t = obs_t / 255.0
                action, _, _ = actor.act(obs_t)
            obs, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            done = terminated or truncated
            time.sleep(0.08)
        print(f"Episode {ep+1}: total_reward = {total_reward:.3f}")
    env.close()


def train_minigrid(args):
    print("============================================================================================")
    device = get_device()

    from torch.utils.tensorboard import SummaryWriter

    sc_gen = ScenarioCreator("src/config/scenario.yaml")
    env = sc_gen.create_env(difficulty=args.difficulty)
    print(f"Loaded environment from ScenarioCreator | Difficulty: {args.difficulty}")

    sample_obs, _ = env.reset()
    obs_dim = int(np.prod(sample_obs.shape))
    print(f"Observation shape: {obs_dim}")
    print(f"Sample obs shape: {obs_dim}, min={sample_obs.min()}, max={sample_obs.max()}")

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_id = sc_gen.config["difficulties"][args.difficulty]["env_id"]
    run_id = f"{env_id}_{args.difficulty}_{timestamp}"

    ckpt_subdir = os.path.join(args.ckpt_dir, run_id)
    tb_dir = os.path.join("tb_logs", run_id)
    os.makedirs(ckpt_subdir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    writer_tb = SummaryWriter(log_dir=tb_dir)

    ckpt_path = os.path.join(ckpt_subdir, "ppo_model.pth")

    step_count = 0
    start_time = time.time()

    print(f"Training started on {device} for {args.total_steps:,} total steps")
    print(f"Environment: {env_id}")
    print(f"Run name: {run_id}")
    print("============================================================================================")

    while step_count < args.total_steps:
        agent.collect_rollouts()
        pi_loss, v_loss, entropy = agent.update()
        step_count += agent.batch_size

        eval_rewards = evaluate_policy(agent, env, episodes=args.eval_episodes)
        avg_r = np.mean(eval_rewards)

        elapsed_min = (time.time() - start_time) / 60

        # ---------------- TensorBoard Scalars ----------------
        writer_tb.add_scalar("reward/avg_eval_reward", avg_r, step_count)
        writer_tb.add_scalar("loss/policy_loss", pi_loss, step_count)
        writer_tb.add_scalar("loss/value_loss", v_loss, step_count)
        writer_tb.add_scalar("loss/entropy", entropy, step_count)

        # Episode stats
        if len(agent.episode_returns) > 0:
            writer_tb.add_scalar("stats/episode_return_mean",
                np.mean(agent.episode_returns[-10:]), step_count)
            writer_tb.add_scalar("stats/episode_length_mean",
                np.mean(agent.episode_lengths[-10:]), step_count)

        # ---------------- TensorBoard Histograms ----------------
        if len(agent.episode_returns) >= 10:
            writer_tb.add_histogram("hist/episode_rewards",
                np.array(agent.episode_returns[-50:]), step_count)
            writer_tb.add_histogram("hist/episode_lengths",
                np.array(agent.episode_lengths[-50:]), step_count)

            # ---------------- TensorBoard Scatter ----------------
            fig = plt.figure()
            plt.scatter(agent.episode_lengths[-50:], agent.episode_returns[-50:], c="green")
            plt.xlabel("Steps")
            plt.ylabel("Reward")
            plt.title("Reward vs Steps")
            writer_tb.add_figure("fig/reward_vs_steps", fig, step_count)
            plt.close(fig)

        if step_count % args.print_interval == 0 or step_count >= args.total_steps:
            print(f"[Steps: {step_count:>7}] | Avg reward: {avg_r:.3f} | Time: {elapsed_min:.2f} min")

        if step_count % args.save_interval == 0 or step_count >= args.total_steps:
            torch.save(agent.ac.state_dict(), ckpt_path)
            print(f"Model checkpoint saved at step {step_count} -> {ckpt_path}")

    env.close()
    writer_tb.close()

    total_time = (time.time() - start_time) / 60
    print("============================================================================================")
    print(f"Training finished | Total time: {total_time:.2f} min")
    print(f"TensorBoard logs: {tb_dir}")
    print(f"Model saved to: {ckpt_path}")
    print("============================================================================================")

    if args.visual_eval:
        visualize_agent(agent, difficulty=args.difficulty, model_path=ckpt_path, episodes=2)


if __name__ == "__main__":
    args = parse_args()
    train_minigrid(args)
