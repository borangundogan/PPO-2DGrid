import numpy as np
import torch
import os
import argparse
import time
import matplotlib.pyplot as plt

from src.scenario_creator.scenario_creator import ScenarioCreator
from src.fomaml import FOMAML
from src.utils.utils import set_seed, get_device

def parse_args():
    parser = argparse.ArgumentParser(description="Train FOMAML on MiniGrid")
    parser.add_argument("--difficulty", type=str, default="medium", 
                        choices=["easy", "medium", "mediumhard", "hard", "hardest"])
    parser.add_argument("--iterations", type=int, default=2000, 
                        help="Total meta-training iterations")
    parser.add_argument("--tasks_per_batch", type=int, default=8,
                        help="Number of tasks (maps) to sample per meta-update")
    parser.add_argument("--k_steps", type=int, default=50,
                        help="Trajectory length")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--render_live", action="store_true", default=True,
                        help="Show maps in a window during training")
    return parser.parse_args()

def train_fomaml():
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    device = get_device(args.device)
    
    # Initialize ScenarioCreator
    sc = ScenarioCreator("src/config/scenario.yaml")
    
    # Checkpoint Directory
    ckpt_dir = os.path.join("checkpoints", "fomaml", f"{args.difficulty}_seed{args.seed}")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Initialize Meta-Learner
    fomaml = FOMAML(
        sc, 
        lr_inner=0.01, 
        lr_outer=3e-4, 
        difficulty=args.difficulty, 
        device=device
    )
    
    print(f"==================================================")
    print(f"[FOMAML] Starting Meta-Training")
    print(f" Difficulty: {args.difficulty}")
    print(f" Batch Size: {args.tasks_per_batch}")
    print(f" Live Render: {'ENABLED' if args.render_live else 'DISABLED'}")
    print(f" Saving to: {ckpt_dir}")
    print(f"==================================================")
    
    # --- Visualization Setup ---
    if args.render_live:
        plt.ion()
        fig, axes = plt.subplots(1, args.tasks_per_batch, figsize=(3 * args.tasks_per_batch, 3))
        if args.tasks_per_batch == 1:
            axes = [axes]
        plt.show() 

    start_time = time.time()
    best_meta_reward = -float('inf') 
    
    for itr in range(1, args.iterations + 1):
        # 1. Sample Task Seeds (Unique tasks for this batch)
        task_seeds = [np.random.randint(0, 100000) for _ in range(args.tasks_per_batch)]
        
        # --- Visualization Loop ---
        if args.render_live and itr % 1 == 0: 
            for i, seed in enumerate(task_seeds):
                temp_env = sc.create_env(args.difficulty, seed=seed)
                temp_env.reset(seed=seed)
                
                # Get pure frame from unwrapped env
                img = temp_env.unwrapped.get_frame() 
                
                axes[i].clear()
                axes[i].imshow(img)
                axes[i].set_title(f"Task {i+1}\nSeed: {seed}", fontsize=9)
                axes[i].axis('off')
                
                temp_env.close()
            
            plt.suptitle(f"Meta-Training Batch | Iteration: {itr}", fontsize=12)
            plt.draw()
            plt.pause(0.5)

        # 2. Perform Meta-Step
        # Returns Loss and Average Reward across the batch
        loss, avg_reward = fomaml.meta_train_step(
            task_seeds, 
            k_support=args.k_steps, 
            k_query=args.k_steps
        )
        
        # 3. Best Model Saving
        if avg_reward > best_meta_reward:
            best_meta_reward = avg_reward
            save_path = os.path.join(ckpt_dir, "best_model.pth")
            torch.save(fomaml.meta_policy.state_dict(), save_path)
            # Optional: Print new record
            # print(f"[*] New Record: {avg_reward:.4f}")
        
        # 4. Logging
        if itr % 10 == 0:
            elapsed = (time.time() - start_time) / 60
            print(f"Iter {itr:>4} | Loss: {loss:.4f} | Rew: {avg_reward:.4f} | Best: {best_meta_reward:.4f} | Time: {elapsed:.1f}m")
            
        # 5. Periodic Checkpoint
        if itr % 100 == 0:
            save_path = os.path.join(ckpt_dir, f"fomaml_iter_{itr}.pth")
            torch.save(fomaml.meta_policy.state_dict(), save_path)
            print(f"[*] Saved periodic model: {save_path}")

    if args.render_live:
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    train_fomaml()