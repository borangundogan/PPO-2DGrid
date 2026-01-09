import numpy as np
import torch
import os
import argparse
import time

# Imports from your src package
from src.scenario_creator.scenario_creator import ScenarioCreator
from src.fomaml import FOMAML
from src.utils.utils import set_seed, get_device

def parse_args():
    parser = argparse.ArgumentParser(description="Train FOMAML on MiniGrid")
    parser.add_argument("--difficulty", type=str, default="medium", 
                        choices=["easy", "medium", "hard", "hardest"])
    parser.add_argument("--iterations", type=int, default=2000, 
                        help="Total meta-training iterations")
    parser.add_argument("--tasks_per_batch", type=int, default=4,
                        help="Number of tasks (maps) to sample per meta-update")
    parser.add_argument("--k_steps", type=int, default=50,
                        help="Trajectory length (Horizon) for Support/Query sets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()

def train_fomaml():
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    device = get_device(args.device)
    
    # Initialize ScenarioCreator (reads from src/config/scenario.yaml)
    # Note: Ensure the path to config is correct relative to root
    sc = ScenarioCreator("src/config/scenario.yaml")
    
    # Checkpoint Directory
    ckpt_dir = os.path.join("checkpoints", "fomaml", f"{args.difficulty}_seed{args.seed}")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Initialize Meta-Learner
    # We use a smaller Inner LR (0.01) and standard Outer LR (3e-4)
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
    print(f" Tasks/Batch: {args.tasks_per_batch}")
    print(f" Device: {device}")
    print(f" Saving to: {ckpt_dir}")
    print(f"==================================================")
    
    start_time = time.time()
    
    for itr in range(1, args.iterations + 1):
        # 1. Sample Task Seeds (The "Task Distribution")
        # We generate random integers to serve as seeds for the ScenarioCreator
        task_seeds = [np.random.randint(0, 100000) for _ in range(args.tasks_per_batch)]
        
        # 2. Perform Meta-Step
        # This runs Inner Loop (Support) -> Outer Loop (Query) -> Meta Update
        loss = fomaml.meta_train_step(
            task_seeds, 
            k_support=args.k_steps, 
            k_query=args.k_steps
        )
        
        # 3. Logging
        if itr % 10 == 0:
            elapsed = (time.time() - start_time) / 60
            print(f"Iter {itr:>4} | Meta-Loss: {loss:.4f} | Time: {elapsed:.1f}m")
            
        # 4. Save Checkpoint
        if itr % 100 == 0:
            save_path = os.path.join(ckpt_dir, f"fomaml_iter_{itr}.pth")
            torch.save(fomaml.meta_policy.state_dict(), save_path)
            print(f"[*] Saved model: {save_path}")

if __name__ == "__main__":
    train_fomaml()