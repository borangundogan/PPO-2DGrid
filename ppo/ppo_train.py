import sys
import os
import time
import argparse
from datetime import datetime
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.utils import set_seed, get_device
from src.ppo import PPO
from src.scenario_creator.scenario_creator import ScenarioCreator
from src.metrics.ppo_metrics import compute_episode_stats

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--update_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--minibatch_size", type=int, default=256)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.05)
    parser.add_argument("--total_steps", type=int, default=300_000)
    parser.add_argument("--save_interval", type=int, default=100_000)
    parser.add_argument("--eval_episodes", type=int, default=3)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--visual_eval", action="store_true")
    parser.add_argument("--print_interval", type=int, default=2048)
    parser.add_argument("--difficulty", type=str, default="easy", choices=["easy", "medium", "mediumhard", "hard", "hardest"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--group_timestamp", type=str, default=None)
    return parser.parse_args()

def evaluate_policy(agent, env, episodes=3, seed=None):
    rewards = []
    steps_list = []
    base_seed = seed if seed is not None else 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)
            
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            with torch.no_grad():
                obs_t = agent._obs_to_tensor(obs)
                action, _, _ = agent.ac.act(obs_t, deterministic=True) 
            obs, reward, terminated, truncated, _ = env.step(action.item())
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs, dtype=np.float32)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        rewards.append(total_reward)
        steps_list.append(steps)
    return rewards, steps_list

def visualize_agent(agent, difficulty="easy", episodes=1):
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
    from src.wrappers.three_action_wrapper import ThreeActionWrapper
    from gymnasium.wrappers import FlattenObservation
    
    env = ImgObsWrapper(env)
    if obs_cfg.get("flatten", False):
        env = FlattenObservation(env)
    env = ThreeActionWrapper(env)

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs, dtype=np.float32)
            with torch.no_grad():
                obs_t = agent._obs_to_tensor(obs)
                action, _, _ = agent.ac.act(obs_t, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            time.sleep(0.08)
    env.close()

def train_minigrid(args):
    set_seed(args.seed)
    device = get_device(args.device)

    sc_gen = ScenarioCreator("src/config/scenario.yaml")
    env = sc_gen.create_env(difficulty=args.difficulty, seed=args.seed)
    eval_env = sc_gen.create_env(args.difficulty, seed=args.seed + 999)

    agent = PPO(
        env=env, lr=args.lr, gamma=args.gamma, lam=args.lam, clip_eps=args.clip_eps,
        update_epochs=args.update_epochs, batch_size=args.batch_size, minibatch_size=args.minibatch_size,
        vf_coef=args.vf_coef, ent_coef=args.ent_coef, device=device
    )
    
    env_id = sc_gen.get_env_id(args.difficulty)
    grid_size_str = sc_gen.get_env_size_str(args.difficulty)
    
    timestamp = args.group_timestamp if args.group_timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{env_id}_{grid_size_str}_{args.difficulty}_{timestamp}"

    ckpt_subdir = os.path.join(args.ckpt_dir, experiment_name, f"seed_{args.seed}")
    tb_dir = os.path.join("tb_logs", experiment_name, f"seed_{args.seed}")
    os.makedirs(ckpt_subdir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    writer_tb = SummaryWriter(log_dir=tb_dir)
    best_model_path = os.path.join(ckpt_subdir, "best_model.pth")

    step_count = 0
    next_save_milestone = args.save_interval
    start_time = time.time()
    best_reward = -float('inf')

    while step_count < args.total_steps:
        last_value = agent.collect_rollouts()
        update_stats = agent.update(last_value)
        step_count += agent.batch_size

        eval_rewards, eval_steps = evaluate_policy(agent, eval_env, episodes=args.eval_episodes, seed=args.seed + 999)
        avg_r = np.mean(eval_rewards)
        avg_s = np.mean(eval_steps)

        if avg_r > best_reward:
            best_reward = avg_r
            torch.save(agent.ac.state_dict(), best_model_path)
            print(f"[*] New best PPO model saved! Reward: {best_reward:.3f} -> {best_model_path}")

        if step_count >= next_save_milestone or step_count >= args.total_steps:
            milestone_k = int(step_count / 1000)
            ckpt_path = os.path.join(ckpt_subdir, f"ppo_model_{milestone_k}k.pth")
            torch.save(agent.ac.state_dict(), ckpt_path)
            next_save_milestone += args.save_interval

        writer_tb.add_scalar("reward/avg_eval_reward", avg_r, step_count)
        writer_tb.add_scalar("loss/policy_loss", update_stats["pi_loss"], step_count)
        writer_tb.add_scalar("loss/value_loss", update_stats["v_loss"], step_count)
        writer_tb.add_scalar("loss/entropy", update_stats["entropy"], step_count)
        writer_tb.add_scalar("diagnostics/kl", update_stats["kl"], step_count)
        writer_tb.add_scalar("diagnostics/clipfrac", update_stats["clipfrac"], step_count)
        writer_tb.add_scalar("diagnostics/gradnorm", update_stats["gradnorm"], step_count)

        if len(agent.episode_returns) > 0:
            stats = compute_episode_stats(agent.episode_returns[-10:], agent.episode_lengths[-10:])
            writer_tb.add_scalar("stats/episode_return_mean", stats["episode_return_mean"], step_count)
            writer_tb.add_scalar("stats/episode_length_mean", stats["episode_length_mean"], step_count)

        if step_count % args.print_interval == 0 or step_count >= args.total_steps:
            elapsed_min = (time.time() - start_time) / 60
            total_loss = update_stats["pi_loss"] + update_stats["v_loss"]
            
            print(f"[{step_count:>7}] R: {avg_r:.3f} | L: {total_loss:.4f} | pi: {update_stats['pi_loss']:.4f} | V: {update_stats['v_loss']:.4f} | Ent: {update_stats['entropy']:.4f} | KL: {update_stats['kl']:.6f} | Steps: {avg_s:.1f} | T: {elapsed_min:.2f}m")
            
            if len(agent.episode_returns) >= 10:
                writer_tb.add_histogram("hist/episode_rewards", np.array(agent.episode_returns[-50:]), step_count)
                writer_tb.add_histogram("hist/episode_lengths", np.array(agent.episode_lengths[-50:]), step_count)
                fig = plt.figure()
                plt.scatter(agent.episode_lengths[-50:], agent.episode_returns[-50:], c="green")
                writer_tb.add_figure("fig/reward_vs_steps", fig, step_count)
                plt.close(fig)

    final_ckpt_path = os.path.join(ckpt_subdir, "ppo_model_final.pth")
    torch.save(agent.ac.state_dict(), final_ckpt_path)

    if args.visual_eval:
        visualize_agent(agent, difficulty=args.difficulty)

if __name__ == "__main__":
    args = parse_args()
    train_minigrid(args)