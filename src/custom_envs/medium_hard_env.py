# src/custom_envs/medium_hard_env.py

from __future__ import annotations

from .base_env import BaseCustomEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal

#TODO update the class
class MediumHardEnv(BaseCustomEnv):
    """
    Mid environment:
    """

    def _gen_grid(self, width, height):
        return 0
