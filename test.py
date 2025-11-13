import os
import time
import argparse
import numpy as np
import torch

from src.actor_critic import ActorCritic
from src.utils import get_device
from src.scenario_creator import ScenarioCreator


# ----------------------------------------------------
# CLI arguments
# ----------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Test a trained PPO agent using ScenarioCreator")
    parser.add_argument("--difficulty", type=str, default="easy",
                        help="Difficulty level (easy, medium, hard)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to trained PPO model (.pth). If not given, loads latest automatically.")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of test episodes to run")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering (useful for headless servers)")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints",
                        help="Root directory for checkpoints (used if model_path not provided)")
    parser.add_argument("--config", type=str, default="config/scenario.yaml",
                        help="Path to scenario configuration file")
    return parser.parse_args()


# ----------------------------------------------------
# Utility: find latest checkpoint
# ----------------------------------------------------
def find_latest_checkpoint(ckpt_root, env_name):
    if not os.path.exists(ckpt_root):
        raise FileNotFoundError(f"Checkpoint root not found: {ckpt_root}")

    candidates = [d for d in os.listdir(ckpt_root) if d.startswith(env_name)]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint folders found for environment: {env_name}")

    def get_model_time(folder):
        model_path = os.path.join(ckpt_root, folder, "ppo_model.pth")
        return os.path.getmtime(model_path) if os.path.exists(model_path) else 0

    latest_dir = sorted(candidates, key=get_model_time)[-1]
    ckpt_path = os.path.join(ckpt_root, latest_dir, "ppo_model.pth")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No model file found in: {ckpt_path}")

    print(f"Found latest checkpoint: {ckpt_path}")
    return ckpt_path


# ----------------------------------------------------
# Test PPO agent
# ----------------------------------------------------
def test_agent(model_path, env, device, episodes=5, render=True):
    obs_sample, _ = env.reset()
    obs_dim = int(np.prod(obs_sample.shape))
    act_dim = env.action_space.n

    # Load trained policy
    actor = ActorCritic(obs_dim, act_dim).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device))
    actor.eval()
    print(f"Loaded model weights from: {model_path}")

    total_rewards = []
    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).view(1, -1)
            obs_t = obs_t / 255.0  # normalize input
            with torch.no_grad():
                action, _, _ = actor.act(obs_t)
            obs, reward, terminated, truncated, _ = env.step(action.item())
            ep_reward += reward
            done = terminated or truncated

            if render:
                time.sleep(0.08)

        total_rewards.append(ep_reward)
        print(f"Episode {ep}/{episodes} | Reward: {ep_reward:.3f}")

    avg_reward = np.mean(total_rewards)
    print("============================================================================================")
    print(f"Average Test Reward: {avg_reward:.3f}")
    print("============================================================================================")
    env.close()


# ----------------------------------------------------
# Entry point
# ----------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    device = get_device()

    # Initialize ScenarioCreator and environment
    sc_gen = ScenarioCreator(args.config)
    env = sc_gen.create_env(difficulty=args.difficulty)
    env_id = sc_gen.get_env_id(args.difficulty)

    # Determine model path automatically if not given
    if args.model_path is None:
        args.model_path = find_latest_checkpoint(args.ckpt_dir, env_id)

    # Run test
    test_agent(
        model_path=args.model_path,
        env=env,
        device=device,
        episodes=args.episodes,
        render=not args.no_render
    )
