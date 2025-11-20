# src/custom_envs/hard_env.py

from __future__ import annotations

from .base_env import BaseCustomEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall, Goal


class HardEnv(BaseCustomEnv):
    """
    Hard difficulty environment:
    - 16x16 grid
    - Vertical wall splits the map into two rooms
    - One open gap in the wall (no door object)
    - Random agent start
    - Random goal placement
    """

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # 1) Outer walls
        self.grid.wall_rect(0, 0, width, height)

        # 2) Vertical separator with one opening
        mid = width // 2             # x = 8
        doorway_y = height // 2      # y = 8

        for y in range(height):
            if y != doorway_y:
                self.grid.set(mid, y, Wall())

        # 3) Place agent randomly
        self.place_agent()

        # 4) Place goal randomly in a free cell
        self.place_obj(Goal())

        # 5) Mission text
        self.mission = "navigate through the gap and reach the goal"