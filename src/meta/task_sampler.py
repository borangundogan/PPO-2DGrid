import numpy as np
from ..scenario_creator import scenario_creator as sc


class MetaTaskSampler:
    def __init__(self, scenario_cfg_path: str, task_names: list[str]):
        self.sc = sc.ScenarioCreator(scenario_cfg_path)
        self.task_names = task_names

    def sample(self, k: int):
        # return list[env], length = k
        envs = []
        for i in range(k):
            diff = np.random.choice(self.task_names)
            env = self.sc.create_env(diff)
            envs.append((diff, env))
        return envs
