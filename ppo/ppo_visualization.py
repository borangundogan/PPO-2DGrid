import sys
import os
import time
import argparse
import glob
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.actor_critic import MLPActorCritic, CNNActorCritic
from src.utils.utils import get_device
from src.scenario_creator.scenario_creator import ScenarioCreator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", type=str, default="easy", choices=["easy", "medium", "mediumhard", "hard", "hardest"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--config", type=str, default="src/config/scenario.yaml")
    parser.add_argument("--no-render", action="store_true", help="Disable PyGame rendering for faster evaluation")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()

def find_latest_checkpoint(ckpt_root, env_name_filter):
    """Recursively search for the most recent 'best_model.pth' matching the environment filter."""
    if not os.path.exists(ckpt_root):
        raise FileNotFoundError(f"Missing directory: {ckpt_root}")

    pattern = os.path.join(ckpt_root, "**", "best_model.pth")
    all_models = glob.glob(pattern, recursive=True)

    if env_name_filter:
        all_models = [m for m in all_models if env_name_filter in m]

    if not all_models:
        raise FileNotFoundError(f"No models found matching filter: {env_name_filter}")

    return max(all_models, key=os.path.getmtime)

def test_agent(model_path, sc_gen, difficulty, device, episodes=10, render=True, seed=None):
    # Override the configuration to enable human rendering if requested
    if render:
        sc_gen.config["difficulties"][difficulty].setdefault("params", {})["render_mode"] = "human"
        
    env = sc_gen.create_env(difficulty, seed=seed)
    
    obs_sample, _ = env.reset(seed=seed)
    if not isinstance(obs_sample, np.ndarray):
        obs_sample = np.array(obs_sample, dtype=np.float32)

    act_dim = env.action_space.n

    # Dynamically select CNN or MLP based on the observation space dimensions
    if obs_sample.ndim == 3:
        use_cnn = True
        policy = CNNActorCritic(obs_sample.shape, act_dim).to(device)
    else:
        use_cnn = False
        policy = MLPActorCritic(int(np.prod(obs_sample.shape)), act_dim).to(device)

    print(f"\n[*] Loading model: {model_path}")
    policy.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    policy.eval()

    rewards = []
    steps_list = [] 

    print(f"[*] Starting Evaluation ({episodes} Episodes)...\n")

    for ep in range(1, episodes + 1):
        # Increment the seed per episode to ensure diverse map topologies
        current_seed = seed + ep if seed else ep
        obs, _ = env.reset(seed=current_seed)
        
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)

        done = False
        ep_reward = 0.0
        steps = 0 

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            obs_t = obs_t.unsqueeze(0) if use_cnn else obs_t.view(1, -1)

            with torch.no_grad():
                # Always use deterministic actions for evaluation
                action, _, _ = policy.act(obs_t, deterministic=True)

            obs, reward, terminated, truncated, _ = env.step(action.item())
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs, dtype=np.float32)

            ep_reward += reward
            steps += 1
            done = terminated or truncated

            # Add a slight delay for visual tracking when rendering is enabled
            if render:
                time.sleep(0.05)

        rewards.append(ep_reward)
        steps_list.append(steps)
        
        # Log individual trajectory performance
        print(f"Episode {ep:>2} | Reward: {ep_reward:.3f} | Steps: {steps}")

    env.close()
    
    # Output aggregated metrics across all evaluated episodes
    print("-" * 40)
    print(f"Average Reward: {np.mean(rewards):.3f}")
    print(f"Average Steps : {np.mean(steps_list):.1f}")
    print("-" * 40)

if __name__ == "__main__":
    args = parse_args()
    device = get_device("auto")
    sc_gen = ScenarioCreator(args.config)

    env_id_filter = sc_gen.get_env_id(args.difficulty)

    model_path = args.model_path
    if not model_path:
        model_path = find_latest_checkpoint(args.ckpt_dir, env_id_filter)

    test_agent(
        model_path=model_path,
        sc_gen=sc_gen,
        difficulty=args.difficulty,
        device=device,
        episodes=args.episodes,
        render=not args.no_render,
        seed=args.seed
    )