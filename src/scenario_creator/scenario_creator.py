# src/scenario_creator.py
import yaml
import numpy as np

from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from gymnasium.envs.registration import register
import gymnasium as gym
import random

class ScenarioCreator:
    """
    ScenarioCreator generates MiniGrid-like environments
    based on difficulty levels and YAML configuration.
    """

    def __init__(self, config_path: str = "config/scenario.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.seed = self.config.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)

    def create_env(self, difficulty: str = "easy"):
        cfg = self.config["difficulties"].get(difficulty)
        if cfg is None:
            raise ValueError(f"Unknown difficulty: {difficulty}")

        env_id = cfg["env_id"]
        env_kwargs = cfg.get("params", {})

        env = gym.make(env_id, **env_kwargs)
        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)
        env.reset(seed=self.seed)
        return env

    def sample_scenarios(self, n: int, difficulty: str):
        """Generate N different randomized seeds for evaluation."""
        return [self.create_env(difficulty) for _ in range(n)]
