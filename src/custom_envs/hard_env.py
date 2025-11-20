# src/custom_envs/hard_env.py

from __future__ import annotations

from .base_env import BaseCustomEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall, Door, Goal


class HardEnv(BaseCustomEnv):
    """
    Hard difficulty environment:
    - 16x16 grid
    - A vertical wall splits the map into two rooms
    - One open door in the middle of the wall (no key needed)
    - Random agent start
    - Random goal
    """

    def _gen_grid(self, width, height):
        # Create grid
        self.grid = Grid(width, height)

        # Outer walls
        self.grid.wall_rect(0, 0, width, height)

        # Add vertical separation wall
        split_x = width // 2  # e.g., 8 for 16x16

        for y in range(1, height - 1):
            self.grid.set(split_x, y, Wall())

        # Add the open door (always unlocked)
        door_y = height // 2  # center
        self.grid.set(split_x, door_y, Door(color='blue', is_locked=False))

        # Random agent
        self.place_agent()

        # Random goal
        self.place_obj(Goal())

        # Mission
        self.mission = "navigate through the door and reach the goal"
