import os
import time
import argparse
import numpy as np
import torch

from src.actor_critic import MLPActorCritic, CNNActorCritic
from src.utils import get_device
from src.scenario_creator.scenario_creator import ScenarioCreator
from minigrid.wrappers import FullyObsWrapper, RGBImgPartialObsWrapper, ImgObsWrapper
from gymnasium.wrappers import FlattenObservation
import gymnasium as gym


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Test PPO model with ScenarioCreator")

    parser.add_argument("--difficulty", type=str, default="easy",
                        help="easy | medium | hard")

    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to .pth model; if not given auto-select latest")

    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--config", type=str, default="src/config/scenario.yaml")

    parser.add_argument("--no-render", action="store_true",
                        help="Disable visual rendering")

    return parser.parse_args()


# ============================================================
# Find latest model automatically
# ============================================================
def find_latest_checkpoint(ckpt_root, env_name):
    if not os.path.exists(ckpt_root):
        raise FileNotFoundError(f"Checkpoint root not found: {ckpt_root}")

    candidates = [d for d in os.listdir(ckpt_root) if d.startswith(env_name)]
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found for: {env_name}")

    def get_mtime(folder):
        model_path = os.path.join(ckpt_root, folder, "ppo_model.pth")
        return os.path.getmtime(model_path) if os.path.exists(model_path) else 0

    latest_folder = sorted(candidates, key=get_mtime)[-1]
    return os.path.join(ckpt_root, latest_folder, "ppo_model.pth")


# ============================================================
# Build environment in HUMAN mode (visualize)
# ============================================================
def build_env_human(sc_gen, difficulty):
    cfg = sc_gen.config["difficulties"][difficulty]
    params = cfg.get("params", {}).copy()
    params["render_mode"] = "human"      # for visualization

    env = gym.make(cfg["env_id"], **params)

    # Apply same wrappers as ScenarioCreator
    obs_cfg = sc_gen.get_observation_params()

    if obs_cfg.get("fully_observable", False):
        env = FullyObsWrapper(env)
    else:
        env = RGBImgPartialObsWrapper(env)

    env = ImgObsWrapper(env)

    if obs_cfg.get("flatten", False):
        env = FlattenObservation(env)
    
    from src.wrappers.three_action_wrapper import ThreeActionWrapper
    env = ThreeActionWrapper(env)

    return env


# ============================================================
# TEST LOOP
# ============================================================
def test_agent(model_path, sc_gen, difficulty, device, episodes=10, render=True):
    print(f"[Test] Loading env: {difficulty}")

    # Create human-render environment
    env = build_env_human(sc_gen, difficulty)
    obs_sample, _ = env.reset()

    # Detect CNN/MLP
    if obs_sample.ndim == 3:
        use_cnn = True
        obs_shape = obs_sample.shape
        obs_dim = None
    else:
        use_cnn = False
        obs_dim = int(np.prod(obs_sample.shape))
        obs_shape = None

    act_dim = env.action_space.n

    # Build correct model architecture
    if use_cnn:
        policy = CNNActorCritic(obs_shape, act_dim).to(device)
        print(f"[Test] Using CNNActorCritic, obs_shape={obs_shape}")
    else:
        policy = MLPActorCritic(obs_dim, act_dim).to(device)
        print(f"[Test] Using MLPActorCritic, obs_dim={obs_dim}")

    # Load weights
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()
    print(f"[Test] Loaded weights: {model_path}")

    # Run test episodes
    rewards = []
    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            if use_cnn:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            else:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).view(1, -1)

            obs_t /= 255.0

            with torch.no_grad():
                action, _, _ = policy.act(obs_t)

            obs, reward, terminated, truncated, _ = env.step(action.item())
            ep_reward += reward
            done = terminated or truncated

            if render:
                time.sleep(0.08)

        rewards.append(ep_reward)
        print(f"[Test] Episode {ep}/{episodes} | Reward: {ep_reward:.3f}")

    print("===============================================================")
    print(f"[Test] Average Reward: {np.mean(rewards):.3f}")
    print("===============================================================")

    env.close()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    args = parse_args()
    device = get_device()

    sc_gen = ScenarioCreator(args.config)
    env_id = sc_gen.get_env_id(args.difficulty)

    # Auto-detect model file
    if args.model_path is None:
        model_path = find_latest_checkpoint(args.ckpt_dir, env_id)
        print(f"[Test] Found latest checkpoint: {model_path}")
    else:
        model_path = args.model_path

    # Run test
    test_agent(
        model_path=model_path,
        sc_gen=sc_gen,
        difficulty=args.difficulty,
        device=device,
        episodes=args.episodes,
        render=not args.no_render
    )
