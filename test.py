import os
import time
import argparse
import numpy as np
import torch

import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper, RGBImgPartialObsWrapper, ImgObsWrapper
from gymnasium.wrappers import FlattenObservation


from src.actor_critic import ActorCritic
from src.utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Test a trained PPO agent on MiniGrid environment")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to trained PPO model (.pth). If not given, loads latest checkpoint automatically.")
    parser.add_argument("--env_name", type=str, default="MiniGrid-Empty-8x8-v0",
                        help="Environment name to test (default: MiniGrid-Empty-8x8-v0)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of test episodes to run")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering (for headless servers)")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints",
                        help="Root directory for checkpoints (used if model_path not provided)")
    return parser.parse_args()

def find_latest_checkpoint(ckpt_root, env_name):
    if not os.path.exists(ckpt_root):
        raise FileNotFoundError(f"Checkpoint root not found: {ckpt_root}")

    candidates = [d for d in os.listdir(ckpt_root) if d.startswith(env_name)]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint folders found for environment: {env_name}")

    def get_model_time(folder):
        model_path = os.path.join(ckpt_root, folder, "ppo_model.pth")
        return os.path.getmtime(model_path) if os.path.exists(model_path) else 0

    candidates = sorted(candidates, key=get_model_time)
    latest_dir = candidates[-1]

    ckpt_path = os.path.join(ckpt_root, latest_dir, "ppo_model.pth")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No model file found in: {ckpt_path}")

    print(f"✅ Found latest checkpoint: {ckpt_path}")
    return ckpt_path


def test_minigrid(model_path=None, env_name="MiniGrid-Empty-8x8-v0", episodes=5, render=True):
    print("============================================================================================")
    print(f"Testing PPO agent on environment: {env_name}")

    device = get_device()

    # Load environment
    env = gym.make(env_name, render_mode="human" if render else None)
    env = FullyObsWrapper(env)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    env = FlattenObservation(env)

    obs_sample, _ = env.reset()
    obs_dim = int(np.prod(obs_sample.shape))
    act_dim = env.action_space.n

    # Load model
    actor = ActorCritic(obs_dim, act_dim).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device))
    actor.eval()

    print(f"Loaded model weights from: {model_path}")
    print("--------------------------------------------------------------------------------------------")

    total_rewards = []
    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).view(1, -1)
            obs_t = obs_t / 255.0  # normalize the input observations !
            with torch.no_grad():
                action, _, _ = actor.act(obs_t)
            obs, reward, terminated, truncated, _ = env.step(action.item())
            ep_reward += reward
            done = terminated or truncated

            if render:
                time.sleep(0.08)

        total_rewards.append(ep_reward)
        print(f"Episode {ep}/{episodes} | Reward: {ep_reward:.3f}")

    env.close()
    avg_reward = np.mean(total_rewards)
    print("============================================================================================")
    print(f"✅ Average Test Reward: {avg_reward:.3f}")
    print("============================================================================================")


# ----------------------------------------------------
# Entry point
# ----------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    # if model path not specified, find latest checkpoint automatically
    if args.model_path is None:
        args.model_path = find_latest_checkpoint(args.ckpt_dir, args.env_name)

    test_minigrid(
        model_path=args.model_path,
        env_name=args.env_name,
        episodes=args.episodes,
        render=not args.no_render
    )
