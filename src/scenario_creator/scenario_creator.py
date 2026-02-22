# src/scenario_creator.py
import os
import yaml
import random
import numpy as np

import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper, RGBImgPartialObsWrapper, ImgObsWrapper
from gymnasium.wrappers import FlattenObservation

from src.wrappers.stuck_penalty_wrapper import StuckPenaltyWrapper
from src.wrappers.three_action_wrapper import ThreeActionWrapper
import src.custom_envs.register

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
        # UPDATED: Uncommented to fix crash in create_env
        self.seed = self.config.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Extract optional sections
        self.global_cfg = self.config.get("global", {})
        self.obs_cfg = self.config.get("observation", {})
        self.rewards_cfg = self.config.get("rewards", {})
        self.logging_cfg = self.config.get("logging", {})

        # NOTE: MERLIN_C2 requires fixed grid size across ALL tasks.
        # Make sure all difficulties use same-size environments (e.g., 8x8).
        # Validate this once here to avoid silent mismatches.
        self._validate_grid_sizes()

    # Validate fixed-size requirement
    def _validate_grid_sizes(self):
        """
        Ensure all envs follow fixed grid size (MERLIN_C2).
        This prevents using MultiRoom or different-size grids by mistake.
        """
        env_ids = [cfg["env_id"] for cfg in self.config["difficulties"].values()]
        # NOTE: MiniGrid env IDs encode size in their name (e.g., 8x8).
        # If sizes mismatch, warn early.
        sizes = set()
        for env_id in env_ids:
            # Try extracting size from pattern "*-8x8-v0"
            if "-" in env_id and "x" in env_id:
                try:
                    size_part = env_id.split("-")[-2]  # e.g., "8x8"
                    sizes.add(size_part)
                except Exception:
                    pass

        if len(sizes) > 1:
            raise ValueError(
                f"[MERLIN_C2 ERROR] Multiple grid sizes detected: {sizes}. "
                f"All environments MUST use the same grid size."
            )
        print(f"[ScenarioCreator] All environments validated as fixed-size: {sizes}")

    # Main environment creation - **kwargs
    def create_env(self, difficulty: str = "easy", seed = None):
        cfg = self.config["difficulties"].get(difficulty)
        if cfg is None:
            raise ValueError(f"Unknown difficulty: {difficulty}")

        env_id = cfg["env_id"] 
        env_kwargs = {**self.global_cfg, **cfg.get("params", {})} # **kwargs

        env = gym.make(env_id, **env_kwargs)

        # Observation type logic
        fully_obs = self.obs_cfg.get("fully_observable", False)
        flatten = self.obs_cfg.get("flatten", False)

        # NOTE: MiniGrid PPO typically uses partial observation (RGBImgPartialObsWrapper)
        # because full observation drastically increases state size.
        # Partial obs is more realistic and aligns with MERLIN assumptions.
        if fully_obs:
            env = FullyObsWrapper(env)
            # print("[ScenarioCreator] Using FullyObsWrapper (Full observation).")
        else:
            env = RGBImgPartialObsWrapper(env)
            # print("[ScenarioCreator] Using RGBImgPartialObsWrapper (Partial observation).")

        # NOTE: ImgObsWrapper converts dict obs → pure image tensor.
        env = ImgObsWrapper(env)

        # NOTE: FlattenObservation should be ON for MLP policies.
        # If using CNN policy, flatten=False.
        if flatten:
            env = FlattenObservation(env)
            # print("[ScenarioCreator] FlattenObservation enabled (MLP input).")
            
        env = ThreeActionWrapper(env)

        if seed is None:
            # UPDATED: This now works because self.seed is defined in __init__
            seed = self.seed
        env.reset(seed=seed)

        return env

    # Evaluation sampling for unseen maps (Meta-RL later)
    def sample_scenarios(self, n: int = 5, difficulty: str = "easy"):
        """Generate multiple randomized environments for evaluation."""
        scenarios = []
        for i in range(n):
            # UPDATED: Ensures deterministic variation based on the master seed
            env = self.create_env(difficulty, seed=self.seed + i)
            # env.reset(seed=self.seed + i)
            scenarios.append(env)

        # NOTE: In PPO-only stage, this is just evaluation sampling.
        # For Meta-RL later, this will be used as task distribution sampling.
        return scenarios

    # Accessors
    def get_env_id(self, difficulty: str) -> str:
        return self.config["difficulties"][difficulty]["env_id"]

    def get_logging_params(self) -> dict:
        return self.logging_cfg

    def get_observation_params(self) -> dict:
        return self.obs_cfg