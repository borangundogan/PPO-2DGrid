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

# --- ENV BUILDER (Adapted for Visualization) ---
def build_env_human(sc_gen, difficulty, seed):
    # Fetch config params and force 'human' render mode
    cfg = sc_gen.config["difficulties"][difficulty]
    params = cfg.get("params", {}).copy()
    params["render_mode"] = "human"
    
    # Create Environment
    import gymnasium as gym
    env = gym.make(cfg["env_id"], **params)

    # Apply Wrapper Chain (Same as training)
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
    
    # Initialize with specific seed
    # FIX: Handling potential Gymnasium API changes for reset
    try:
        env.reset(seed=seed)
    except TypeError:
        # Fallback for older/newer Gym versions if kwargs differ
        env.seed(seed)
        env.reset()
        
    return env

# --- SINGLE EPISODE VISUALIZER ---
def run_visual_episode(env, policy, device, title="Episode", deterministic=False):
    print(f"--- {title} ---")
    
    # Reset Environment
    obs, _ = env.reset()
    
    done = False
    total_reward = 0
    step_count = 0
    
    while not done:
        env.render() # Show window
        time.sleep(0.05) # Slow down for human eye

        # Process Observation
        obs_np = np.array(obs)
        
        # Determine if input is Image (CNN) or Flat (MLP)
        if obs_np.ndim == 3: # (H, W, C) -> Image
            # Convert to PyTorch Tensor: (B, C, H, W)
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            obs_t = obs_t.permute(0, 3, 1, 2) # Rearrange channels
        else: # (N,) -> Flat Vector
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).view(1, -1)

        # Select Action
        with torch.no_grad():
            # Use deterministic=True for testing/visualization (Greedy)
            action, _, _ = policy.act(obs_t, deterministic=deterministic)

        # Step Environment
        obs, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += reward
        step_count += 1
        
        # Force stop if too long (optional safety)
        if step_count >= 100: 
            truncated = True
            
        done = terminated or truncated

    print(f"Result: Reward={total_reward:.3f} | Steps={step_count}")
    time.sleep(1.0) # Pause to see result
    return total_reward

# --- MAIN FUNCTION ---
def main():
    parser = argparse.ArgumentParser(description="Visualize FOMAML Adaptation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained .pth model")
    parser.add_argument("--difficulty", type=str, default="medium")
    parser.add_argument("--lr_inner", type=float, default=0.001, help="Adaptation learning rate")
    parser.add_argument("--num_tasks", type=int, default=3, help="How many different maps to visualize")
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    device = get_device("auto")
    sc = ScenarioCreator("src/config/scenario.yaml")
    
    # 1. Load Model Architecture (Meta-Policy)
    # Create a dummy env to infer input shapes
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

    # Initialize Helper (for compute_loss & data collection)
    fomaml_helper = FOMAML(sc, device=device, difficulty=args.difficulty)
    
    # Load weights into helper's policy as well (just in case)
    fomaml_helper.meta_policy.load_state_dict(meta_policy.state_dict())

    # --- TASK LOOP ---
    for i in range(args.num_tasks):
        task_seed = args.seed + i
        print(f"\n{'='*40}")
        print(f"VISUALIZING TASK {i+1}/{args.num_tasks} (Seed: {task_seed})")
        print(f"{'='*40}")

        # A. PRE-UPDATE (Zero-Shot) VISUALIZATION
        # -------------------------------------------------
        print(">> Phase 1: Pre-Update (Zero-Shot Performance)")
        env_human = build_env_human(sc, args.difficulty, seed=task_seed)
        run_visual_episode(env_human, meta_policy, device, title="Pre-Update", deterministic=False)
        env_human.close()

        # B. ADAPTATION (Background Fast Learning)
        # -------------------------------------------------
        print(">> Phase 2: Running Inner Loop Adaptation (Thinking...)")
        
        # Use a non-visual (fast) env for adaptation
        # CRITICAL: Must use SAME Seed to learn the SAME map
        env_fast = sc.create_env(args.difficulty, seed=task_seed)
        
        # Create Fast Policy (Clone)
        fast_policy = deepcopy(meta_policy)
        fast_policy.train() # Enable train mode for gradients
        
        # Use same optimizer settings as training
        inner_optim = optim.SGD(fast_policy.parameters(), lr=args.lr_inner)
        
        # Collect Support Data (Inner Loop Experience)
        # We use the helper method from FOMAML class
        support_data = fomaml_helper.collect_trajectory(env_fast, meta_policy, steps=100)
        
        # Compute Loss & Update
        # Note: We use fomaml_helper.compute_loss but applied to our local fast_policy
        loss = fomaml_helper.compute_loss(support_data, fast_policy)
        
        inner_optim.zero_grad()
        loss.backward()
        # Optional: Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(fast_policy.parameters(), max_norm=0.5)
        inner_optim.step()
        
        env_fast.close()
        print(">> Adaptation Complete!")

        # C. POST-UPDATE (Few-Shot) VISUALIZATION
        # -------------------------------------------------
        print(">> Phase 3: Post-Update (Adapted Performance)")
        
        # Re-open visual env (SAME Seed)
        env_human = build_env_human(sc, args.difficulty, seed=task_seed)
        
        # Now use the 'fast_policy' (Adapted Brain)
        fast_policy.eval() 
        run_visual_episode(env_human, fast_policy, device, title="Post-Update", deterministic=True)
        env_human.close()

if __name__ == "__main__":
    main()