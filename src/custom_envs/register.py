# src/custom_envs/register.py

import gymnasium as gym

# Import env classes so gym can instantiate them
from .easy_env import EasyEnv
from .medium_env import MediumEnv
from .hard_env import HardEnv
from .hardest_env import HardestEnv



# Gym environment registration
gym.register(
    id="MERLIN-Easy-16x16-v0",
    entry_point="src.custom_envs.easy_env:EasyEnv",
)

gym.register(
    id="MERLIN-Medium-16x16-v0",
    entry_point="src.custom_envs.medium_env:MediumEnv",
)

gym.register(
    id="MERLIN-Hard-16x16-v0",
    entry_point="src.custom_envs.hard_env:HardEnv",
)

gym.register(
    id="MERLIN-Hardest-16x16-v0",
    entry_point="src.custom_envs.hardest_env:HardestEnv",
)
