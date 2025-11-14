# src/scenario_creator.py
import os
import yaml
import random
import numpy as np
import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper, RGBImgPartialObsWrapper, ImgObsWrapper
from gymnasium.wrappers import FlattenObservation


class ScenarioCreator:
    """
    ScenarioCreator dynamically generates MiniGrid environments
    based on difficulty levels and YAML configuration.
    Automatically configures wrappers (Full/Partial Obs, Flatten, etc.)
    """

    def __init__(self, config_path: str = "config/scenario.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Scenario file not found: {config_path}")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # --- Global seed for reproducibility ---
        self.seed = self.config.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # --- Extract optional sections ---
        self.global_cfg = self.config.get("global", {})
        self.obs_cfg = self.config.get("observation", {})
        self.rewards_cfg = self.config.get("rewards", {})
        self.logging_cfg = self.config.get("logging", {})

    # ----------------------------------------------------------
    # Main environment creation
    # ----------------------------------------------------------
    def create_env(self, difficulty: str = "easy"):
        cfg = self.config["difficulties"].get(difficulty)
        if cfg is None:
            raise ValueError(f"Unknown difficulty: {difficulty}")

        env_id = cfg["env_id"]
        env_kwargs = {**self.global_cfg, **cfg.get("params", {})}

        env = gym.make(env_id, **env_kwargs)

        # --- Observation type logic ---
        fully_obs = self.obs_cfg.get("fully_observable", False)
        normalize = self.obs_cfg.get("normalize", True)
        flatten = self.obs_cfg.get("flatten", False)

        # Apply wrappers dynamically
        if fully_obs:
            env = FullyObsWrapper(env)
            print("[ScenarioCreator] Using FullyObsWrapper (Full observation).")
        else:
            env = RGBImgPartialObsWrapper(env)
            print("[ScenarioCreator] Using RGBImgPartialObsWrapper (Partial observation).")

        env = ImgObsWrapper(env)

        if flatten:
            env = FlattenObservation(env)
            print("[ScenarioCreator] FlattenObservation enabled (MLP input).")

        env.reset(seed=self.seed)
        return env

    # ----------------------------------------------------------
    # Evaluation sampling (for Meta-RL / generalization tests)
    # ----------------------------------------------------------
    def sample_scenarios(self, n: int = 5, difficulty: str = "easy"):
        """Generate multiple randomized environments for evaluation."""
        scenarios = []
        for i in range(n):
            env = self.create_env(difficulty)
            env.reset(seed=self.seed + i)
            scenarios.append(env)
        return scenarios

    # ----------------------------------------------------------
    # Accessors for configs
    # ----------------------------------------------------------
    def get_env_id(self, difficulty: str) -> str:
        return self.config["difficulties"][difficulty]["env_id"]

    def get_logging_params(self) -> dict:
        return self.logging_cfg

    def get_observation_params(self) -> dict:
        return self.obs_cfg
