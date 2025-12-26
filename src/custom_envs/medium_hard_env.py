# src/custom_envs/medium_hard_env.py

from __future__ import annotations
import random

from .base_env import BaseCustomEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Wall


class MediumHardEnv(BaseCustomEnv):
    """
    Medium-Hard environment (Scattered Pillars):
    - 16x16 grid
    - Random agent start
    - Random goal position
    - Scattered single-block walls (Pillars) acting as obstacles.
    - Bridges the gap between empty Medium and walled Hard envs.
    """

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        num_obstacles = random.randint(20, 30)

        for _ in range(num_obstacles):

            self.place_obj(Wall(), max_tries=100)

        self.place_agent()

        self.place_obj(Goal())

        self.mission = "avoid pillars and reach the goal"