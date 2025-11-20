import os
import sys
import time
import random

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

import src.custom_envs.register
from src.scenario_creator.scenario_creator import ScenarioCreator


def visualize_all_envs():
    config_path = os.path.join(ROOT, "src", "config", "scenario.yaml")
    sc = ScenarioCreator(config_path)

    difficulties = ["easy", "medium", "hard", "hardest"]

    for diff in difficulties:
        print(f"\n=== VISUALIZING: {diff.upper()} ===")

        # Render mode MUST be human for MiniGrid visual output
        env_id = sc.get_env_id(diff)
        env = sc.create_env(diff)

        # MiniGrid expects render_mode="human" for window
        env.close()  # close wrapped version
        import gymnasium as gym
        env = gym.make(env_id, render_mode="human")

        obs, info = env.reset(seed=123)

        for step in range(20):
            a = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(a)

            time.sleep(0.2)  # slow down so you can watch movement
            if terminated or truncated:
                break

        env.close()


if __name__ == "__main__":
    visualize_all_envs()
