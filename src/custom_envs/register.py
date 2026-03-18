# src/custom_envs/register.py

import gymnasium as gym

from .easy_env import EasyEnv
from .medium_env import MediumEnv
from .hard_env import HardEnv
from .hardest_env import HardestEnv
from .medium_hard_env import MediumHardEnv

gym.register(
    id="MERLIN-Easy-v0",
    entry_point="src.custom_envs.easy_env:EasyEnv",
)

gym.register(
    id="MERLIN-Medium-v0",
    entry_point="src.custom_envs.medium_env:MediumEnv",
)

gym.register(
    id="MERLIN-MediumHard-v0",
    entry_point="src.custom_envs.medium_hard_env:MediumHardEnv",
)

gym.register(
    id="MERLIN-Hard-v0",
    entry_point="src.custom_envs.hard_env:HardEnv",
)

gym.register(
    id="MERLIN-Hardest-v0",
    entry_point="src.custom_envs.hardest_env:HardestEnv",
)