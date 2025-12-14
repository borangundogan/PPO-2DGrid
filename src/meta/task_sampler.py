# src/meta/task_sampler.py

import numpy as np
from typing import List, Dict
from src.scenario_creator.scenario_creator import ScenarioCreator


class MetaTaskSampler:
    """
    Samples tasks (environments) for Meta-RL.

    Each sampled task corresponds to:
      - one difficulty family (easy/medium/hard/hardest)
      - one concrete environment instance with its own seed

    This is used by the outer loop to create a batch of tasks.
    """

    def __init__(
        self,
        scenario_cfg_path: str,
        task_names: List[str],
        base_seed: int = 0,
    ):
        self.sc = ScenarioCreator(scenario_cfg_path)
        self.task_names = task_names
        self.base_seed = base_seed
        self._counter = 0  # ensures different seeds per sample

    def sample(self, k: int) -> List[Dict]:
        """
        Sample k tasks.

        Returns:
            List of dicts:
              {
                "task_name": str,
                "env": gym.Env,
                "seed": int
              }
        """
        tasks = []

        for _ in range(k):
            task_name = np.random.choice(self.task_names)

            seed = self.base_seed + self._counter
            self._counter += 1

            env = self.sc.create_env(
                difficulty=task_name,
                seed=seed
            )

            tasks.append({
                "task_name": task_name,
                "env": env,
                "seed": seed
            })

        return tasks
