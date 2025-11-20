import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

import src.custom_envs.register  # triggers register.py
from src.scenario_creator.scenario_creator import ScenarioCreator


def test_scenarios():
    config_path = os.path.join(ROOT, "src", "config", "scenario.yaml")
    print("-> USING CONFIG:", config_path)
    print("-> EXISTS:", os.path.exists(config_path))

    sc = ScenarioCreator(os.path.join(ROOT, "src", "config", "scenario.yaml"))

    for diff in ["easy", "medium", "hard", "hardest"]:
        print(f"\n--- Testing difficulty: {diff} ---")

        env = sc.create_env(diff)
        obs, info = env.reset(seed=123)

        print("Env loaded:", sc.get_env_id(diff))
        print("Observation shape:", getattr(obs, "shape", None))
        print("Action space:", env.action_space)

        for i in range(5):
            a = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(a)
            print(f"Step {i}: reward={reward}, terminated={terminated}, truncated={truncated}")

        print("✓ Scenario OK")


if __name__ == "__main__":
    test_scenarios()
