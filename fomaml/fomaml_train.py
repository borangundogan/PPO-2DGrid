import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import argparse
import time
import matplotlib.pyplot as plt
from datetime import datetime

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
    parser.add_argument("--k_steps", type=int, default=100,
                        help="Trajectory length")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    
    parser.add_argument("--render_live", action="store_true", default=False,
                        help="Open a window to show the agent/maps during training")
    parser.add_argument("--plot_curves", action="store_true", default=False,
                        help="Open a live window to plot Reward and Steps")
                        
    return parser.parse_args()

def train_fomaml():
    args = parse_args()
    
    set_seed(args.seed)
    device = get_device(args.device)
    
    sc = ScenarioCreator("src/config/scenario.yaml")
    
    env_id = sc.get_env_id(args.difficulty)
    grid_size_str = "UnknownSize"
    parts = env_id.split('-')
    for p in parts:
        if 'x' in p and p[0].isdigit(): 
            grid_size_str = p
            break
            
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = f"MERLIN-{args.difficulty.capitalize()}-{grid_size_str}-v0_FOMAML_{timestamp}"
    ckpt_dir = os.path.join("checkpoints", project_name, f"seed_{args.seed}")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    fomaml = FOMAML(
        sc, 
        lr_inner=0.01, 
        lr_outer=3e-4, 
        difficulty=args.difficulty, 
        device=device
    )
    
    print(f"==================================================")
    print(f"[FOMAML] Starting Meta-Training")
    print(f" Project      : {project_name}")
    print(f" Difficulty   : {args.difficulty}")
    print(f" Env ID       : {env_id}")
    print(f" Seed         : {args.seed}")
    print(f" Saving to    : {ckpt_dir}")
    print(f" Live Map     : {'ON' if args.render_live else 'OFF'}")
    print(f" Live Curves  : {'ON' if args.plot_curves else 'OFF'}")
    print(f"==================================================")
    
    if args.render_live or args.plot_curves:
        plt.ion()
    
    fig_map, axes_map = None, None
    if args.render_live:
        fig_map, axes_map = plt.subplots(1, args.tasks_per_batch, figsize=(3 * args.tasks_per_batch, 3))
        if args.tasks_per_batch == 1: axes_map = [axes_map]
        fig_map.canvas.manager.set_window_title("Live Agent View")

    fig_metrics, axes_metrics = None, None
    if args.plot_curves:
        fig_metrics, axes_metrics = plt.subplots(2, 1, figsize=(6, 8))
        plt.tight_layout(pad=3.0)
        fig_metrics.canvas.manager.set_window_title("Live Training Metrics")
        
    if args.render_live or args.plot_curves:
        plt.show()

    start_time = time.time()
    best_meta_reward = -float('inf') 
    
    history = {
        "iter": [],
        "loss": [],
        "rew": [],
        "steps": []
    }
    
    for itr in range(1, args.iterations + 1):
        task_seeds = [int(s) for s in np.random.choice(range(100000), size=args.tasks_per_batch, replace=False)]
        
        if args.render_live and itr % 1 == 0: 
            for i, seed in enumerate(task_seeds):
                temp_env = sc.create_env(args.difficulty, seed=seed)
                temp_env.reset(seed=int(seed))
                img = temp_env.unwrapped.get_frame() 
                axes_map[i].clear()
                axes_map[i].imshow(img)
                axes_map[i].set_title(f"Task {i+1}", fontsize=8)
                axes_map[i].axis('off')
                temp_env.close()
            fig_map.suptitle(f"Iter: {itr}", fontsize=12)
            fig_map.canvas.draw()
            fig_map.canvas.flush_events()

        loss, avg_reward, steps = fomaml.meta_train_step(
            task_seeds, 
            k_support=args.k_steps, 
            k_query=args.k_steps
        )
        
        history["iter"].append(itr)
        history["loss"].append(loss)
        history["rew"].append(avg_reward)
        history["steps"].append(steps)
        
        if avg_reward > best_meta_reward:
            best_meta_reward = avg_reward
            save_path = os.path.join(ckpt_dir, "best_model.pth")
            torch.save(fomaml.meta_policy.state_dict(), save_path)
            print(f"[*] New Best Model Saved (Rew: {best_meta_reward:.4f})")
        
        if itr % 10 == 0:
            elapsed = (time.time() - start_time) / 60
            print(f"Iter {itr:>4} | Loss: {loss:.4f} | Rew: {avg_reward:.4f} | Steps: {steps:.1f} | Best: {best_meta_reward:.4f} | Time: {elapsed:.1f}m")
            
        if itr % 100 == 0: 
            save_path = os.path.join(ckpt_dir, f"fomaml_iter_{itr}.pth")
            torch.save(fomaml.meta_policy.state_dict(), save_path)
            
            temp_fig, temp_ax = plt.subplots(2, 1, figsize=(10, 8))
            
            temp_ax[0].plot(history["iter"], history["rew"], color='green', label='Avg Reward')
            temp_ax[0].set_title("Meta-Test Reward")
            temp_ax[0].set_ylabel("Reward (0-1)")
            temp_ax[0].grid(True, alpha=0.3)
            
            temp_ax[1].plot(history["iter"], history["steps"], color='blue', label='Steps to Goal')
            temp_ax[1].set_title("Navigation Efficiency")
            temp_ax[1].set_ylabel("Steps")
            temp_ax[1].set_xlabel("Iterations")
            temp_ax[1].grid(True, alpha=0.3)
            
            plot_path = os.path.join(ckpt_dir, "training_curves.png")
            temp_fig.savefig(plot_path)
            plt.close(temp_fig) 
            
            if args.plot_curves:
                ax = axes_metrics
                ax[0].cla(); ax[1].cla()
                
                ax[0].plot(history["iter"], history["rew"], 'g-')
                ax[0].set_title("Reward")
                ax[0].grid(True, alpha=0.3)
                
                ax[1].plot(history["iter"], history["steps"], 'b-')
                ax[1].set_title("Steps")
                ax[1].grid(True, alpha=0.3)
                
                fig_metrics.canvas.draw()
                fig_metrics.canvas.flush_events()
            
            print(f"[*] Saved training curves to: {plot_path}")

    if args.render_live or args.plot_curves:
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    train_fomaml()