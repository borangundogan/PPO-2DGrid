import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import numpy as np
import torch
import torch.optim as optim
from copy import deepcopy

from src.actor_critic import MLPActorCritic, CNNActorCritic
from src.utils.utils import get_device, set_seed
from src.scenario_creator.scenario_creator import ScenarioCreator
from src.fomaml import FOMAML 

def build_env_human(sc_gen, difficulty, seed):
    cfg = sc_gen.config["difficulties"][difficulty]
    
    global_cfg = sc_gen.config.get("global", {})
    params = {**global_cfg, **cfg.get("params", {})}
    params["render_mode"] = "human"
    
    import gymnasium as gym
    env = gym.make(cfg["env_id"], **params)

    from minigrid.wrappers import FullyObsWrapper, RGBImgPartialObsWrapper, ImgObsWrapper
    from gymnasium.wrappers import FlattenObservation
    from src.wrappers.three_action_wrapper import ThreeActionWrapper

    obs_cfg = sc_gen.get_observation_params()

    if obs_cfg.get("fully_observable", False):
        env = FullyObsWrapper(env)
    else:
        env = RGBImgPartialObsWrapper(env)

    env = ImgObsWrapper(env)

    if obs_cfg.get("flatten", False):
        env = FlattenObservation(env)

    env = ThreeActionWrapper(env)
    
    try:
        env.reset(seed=seed)
    except TypeError:
        env.seed(seed)
        env.reset()
        
    return env

def run_visual_episode(env, policy, device, title="Episode", deterministic=False):
    print(f"--- {title} ---")
    
    obs, _ = env.reset()
    
    done = False
    total_reward = 0
    step_count = 0
    
    while not done:
        env.render() 
        time.sleep(0.05) 

        obs_np = np.array(obs)
        
        if obs_np.ndim == 3: 
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        else: 
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).view(1, -1)

        with torch.no_grad():
            action, _, _ = policy.act(obs_t, deterministic=deterministic, epsilon=0.1)

        obs, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += reward
        step_count += 1
        
        if step_count >= 256: 
            truncated = True
            
        done = terminated or truncated

    print(f"Result: Reward={total_reward:.3f} | Steps={step_count}")
    time.sleep(1.0) 
    return total_reward

def main():
    parser = argparse.ArgumentParser(description="Visualize FOMAML Adaptation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained .pth model")
    parser.add_argument("--difficulty", type=str, default="medium")
    parser.add_argument("--lr_inner", type=float, default=0.001, help="Adaptation learning rate")
    parser.add_argument("--num_tasks", type=int, default=3, help="How many different maps to visualize")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--k_support", type=int, default=256, help="Trajectory length for adaptation")
    args = parser.parse_args()

    device = get_device("auto")
    sc = ScenarioCreator("src/config/scenario.yaml")
    
    dummy_env = sc.create_env(args.difficulty, seed=42)
    obs_sample, _ = dummy_env.reset()
    act_dim = dummy_env.action_space.n
    
    if np.array(obs_sample).ndim == 3:
        meta_policy = CNNActorCritic(np.array(obs_sample).shape, act_dim).to(device)
    else:
        meta_policy = MLPActorCritic(int(np.prod(np.array(obs_sample).shape)), act_dim).to(device)
    
    print(f"[Visualizer] Loading model: {args.model_path}")
    meta_policy.load_state_dict(torch.load(args.model_path, map_location=device))
    meta_policy.eval()

    fomaml_helper = FOMAML(sc, device=device, difficulty=args.difficulty)
    fomaml_helper.meta_policy.load_state_dict(meta_policy.state_dict())

    for i in range(args.num_tasks):
        task_seed = args.seed + i
        print(f"\n{'='*40}")
        print(f"VISUALIZING TASK {i+1}/{args.num_tasks} (Seed: {task_seed})")
        print(f"{'='*40}")

        print(">> Phase 1: Pre-Update (Zero-Shot Performance)")
        env_human = build_env_human(sc, args.difficulty, seed=task_seed)
        run_visual_episode(env_human, meta_policy, device, title="Pre-Update", deterministic=True)
        env_human.close()

        print(">> Phase 2: Running Inner Loop Adaptation (Thinking...)")
        
        env_fast = sc.create_env(args.difficulty, seed=task_seed)
        
        fast_policy = deepcopy(meta_policy)
        fast_policy.train() 
        
        inner_optim = optim.SGD(fast_policy.parameters(), lr=args.lr_inner)
        
        support_data = fomaml_helper.collect_trajectory(env_fast, fast_policy, steps=args.k_support)
        
        loss = fomaml_helper.compute_loss(support_data, fast_policy)
        
        inner_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fast_policy.parameters(), max_norm=0.5)
        inner_optim.step()
        
        env_fast.close()
        print(">> Adaptation Complete!")

        print(">> Phase 3: Post-Update (Adapted Performance)")
        
        env_human = build_env_human(sc, args.difficulty, seed=task_seed)
        
        fast_policy.eval() 
        run_visual_episode(env_human, fast_policy, device, title="Post-Update", deterministic=True)
        env_human.close()

if __name__ == "__main__":
    main()