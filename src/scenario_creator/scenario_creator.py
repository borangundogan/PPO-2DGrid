import os
import yaml
import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper, RGBImgPartialObsWrapper, ImgObsWrapper
from gymnasium.wrappers import FlattenObservation

from src.wrappers.three_action_wrapper import ThreeActionWrapper
import src.custom_envs.register

class ScenarioCreator:
    def __init__(self, config_path: str = "src/config/scenario.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.seed = self.config.get("seed", 42)
        self.global_cfg = self.config.get("global", {})
        self.obs_cfg = self.config.get("observation", {})
        self.rewards_cfg = self.config.get("rewards", {})
        self.logging_cfg = self.config.get("logging", {})

        self._validate_grid_sizes()

    def _validate_grid_sizes(self):
        sizes = {
            cfg["env_id"].split("-")[-2]
            for cfg in self.config["difficulties"].values()
            if "-" in cfg["env_id"] and "x" in cfg["env_id"]
        }
        if len(sizes) > 1:
            raise ValueError(f"Multiple grid sizes detected: {sizes}")

    def create_env(self, difficulty: str = "easy", seed=None):
        cfg = self.config["difficulties"].get(difficulty)
        if not cfg:
            raise ValueError(f"Unknown difficulty: {difficulty}")

        env_id = cfg["env_id"]
        env_kwargs = {**self.global_cfg, **cfg.get("params", {})}
        
        env = gym.make(env_id, **env_kwargs)

        if self.obs_cfg.get("fully_observable", False):
            env = FullyObsWrapper(env)
        else:
            env = RGBImgPartialObsWrapper(env)

        env = ImgObsWrapper(env)

        if self.obs_cfg.get("flatten", False):
            env = FlattenObservation(env)
            
        env = ThreeActionWrapper(env)

        return env

    def sample_scenarios(self, n: int = 5, difficulty: str = "easy"):
        return [self.create_env(difficulty) for _ in range(n)]

    def get_env_id(self, difficulty: str) -> str:
        return self.config["difficulties"][difficulty]["env_id"]

    def get_logging_params(self) -> dict:
        return self.logging_cfg

    def get_observation_params(self) -> dict:
        return self.obs_cfg