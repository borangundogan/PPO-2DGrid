# src/custom_envs/medium_env.py

from __future__ import annotations

from .base_env import BaseCustomEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal


class MediumEnv(BaseCustomEnv):
    """
    Medium difficulty:
    - 16x16 empty grid
    - Random agent start position
    - Random goal position
    - No internal walls
    """

    def _gen_grid(self, width, height):
        # Create empty grid
        self.grid = Grid(width, height)

        # Surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Randomize agent (MiniGrid's built-in function)
        self.place_agent()

        # Random goal placement in the environment
        self.place_obj(Goal())

        # Mission screen
        self.mission = "navigate to the randomly placed goal"
