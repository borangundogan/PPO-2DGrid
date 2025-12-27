import os
import time
import argparse
import glob
import numpy as np
import torch
import gymnasium as gym

from src.actor_critic import MLPActorCritic, CNNActorCritic
from src.utils import get_device
from src.scenario_creator.scenario_creator import ScenarioCreator
from minigrid.wrappers import FullyObsWrapper, RGBImgPartialObsWrapper, ImgObsWrapper
from gymnasium.wrappers import FlattenObservation

# CLI
def parse_args():
    parser = argparse.ArgumentParser(description="Test PPO model with ScenarioCreator")

    parser.add_argument("--difficulty", type=str, default="easy",
                        choices=["easy", "medium", "mediumhard" ,"hard", "hardest"])

    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to .pth model; if not given auto-select latest")

    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--config", type=str, default="src/config/scenario.yaml")

    parser.add_argument("--no-render", action="store_true",
                        help="Disable visual rendering")

    parser.add_argument("--seed", type=int, default=123,
                    help="Random seed for evaluation environment")

    return parser.parse_args()


def find_latest_checkpoint(ckpt_root, env_name_filter=None):
    if not os.path.exists(ckpt_root):
        raise FileNotFoundError(f"Checkpoint root not found: {ckpt_root}")

    search_pattern = os.path.join(ckpt_root, "**", "ppo_model.pth")
    all_models = glob.glob(search_pattern, recursive=True)

    if not all_models:
        raise FileNotFoundError(f"No 'ppo_model.pth' found in {ckpt_root} or its subdirectories.")

    if env_name_filter:
        all_models = [m for m in all_models if env_name_filter in m]
        if not all_models:
            raise FileNotFoundError(f"No models found matching filter: {env_name_filter}")

    latest_model = max(all_models, key=os.path.getmtime)
    
    return latest_model


# Build environment in HUMAN mode (visualize)
def build_env_human(sc_gen, difficulty):
    cfg = sc_gen.config["difficulties"][difficulty]
    params = cfg.get("params", {}).copy()
    params["render_mode"] = "human"

    env = gym.make(cfg["env_id"], **params)

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


# TEST LOOP
def test_agent(model_path, sc_gen, difficulty, device, episodes=10, render=True, seed=None):
    print(f"[Test] Loading env: {difficulty}")
    env = build_env_human(sc_gen, difficulty)
    
    base_seed = seed if seed is not None else 0

    obs_sample, _ = env.reset(seed=base_seed)
    obs_sample = np.array(obs_sample, dtype=np.float32)

    if obs_sample.ndim == 3:
        use_cnn = True
        obs_shape = obs_sample.shape
        obs_dim = None
    else:
        use_cnn = False
        obs_dim = int(np.prod(obs_sample.shape))
        obs_shape = None

    act_dim = env.action_space.n

    if use_cnn:
        policy = CNNActorCritic(obs_shape, act_dim).to(device)
        print(f"[Test] Using CNNActorCritic, obs_shape={obs_shape}")
    else:
        policy = MLPActorCritic(obs_dim, act_dim).to(device)
        print(f"[Test] Using MLPActorCritic, obs_dim={obs_dim}")

    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()
    print(f"[Test] Loaded weights: {model_path}")

    rewards = []

    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=base_seed + ep)
        obs = np.array(obs, dtype=np.float32)
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
            obs = np.array(obs, dtype=np.float32)
            ep_reward += reward
            done = terminated or truncated

            if render:
                time.sleep(0.05)

        rewards.append(ep_reward)
        print(f"[Test] Episode {ep}/{episodes} | Reward: {ep_reward:.3f}")

    print("===============================================================")
    print(f"[Test] Average Reward: {np.mean(rewards):.3f}")
    print("===============================================================")

    env.close()


if __name__ == "__main__":
    args = parse_args()
    device = get_device("auto")

    sc_gen = ScenarioCreator(args.config)

    env_id_filter = sc_gen.config["difficulties"][args.difficulty]["env_id"]

    if args.model_path is None:
        try:
            model_path = find_latest_checkpoint(args.ckpt_dir, env_name_filter=None)
            print(f"[Test] Auto-selected latest checkpoint: {model_path}")
        except FileNotFoundError as e:
            print(f"[Error] {e}")
            exit(1)
    else:
        model_path = args.model_path

    test_agent(
        model_path=model_path,
        sc_gen=sc_gen,
        difficulty=args.difficulty,
        device=device,
        episodes=args.episodes,
        render=not args.no_render,
        seed=args.seed
    )