# src/custom_envs/veryhard_env.py

from __future__ import annotations

import random
from .base_env import BaseCustomEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall, Door, Goal


class HardestEnv(BaseCustomEnv):
    """
    Very hard environment:
    - 16x16 FourRooms-style maze
    - Two open doors connecting regions
    - 5-10 random obstacles
    - Random agent start
    - Random goal
    """

    def _gen_grid(self, width, height):
        # Create grid
        self.grid = Grid(width, height)

        # Outer walls
        self.grid.wall_rect(0, 0, width, height)

        # 1) Create Four-Rooms-style layout
        mid_x = width // 2   # 8
        mid_y = height // 2  # 8

        # Vertical separator
        for y in range(1, height - 1):
            self.grid.set(mid_x, y, Wall())

        # Horizontal separator
        for x in range(1, width - 1):
            self.grid.set(x, mid_y, Wall())

        # 2) Add open doors
        # Vertical wall door
        door_y = random.randint(2, height - 3)
        self.grid.set(mid_x, door_y, Door(color='blue', is_locked=False))

        # Horizontal wall door
        door_x = random.randint(2, width - 3)
        self.grid.set(door_x, mid_y, Door(color='green', is_locked=False))

        # 3) Add random obstacles (5–10)
        num_obstacles = random.randint(5, 10)
        for _ in range(num_obstacles):
            # Try random placements until we find an empty spot
            x = random.randint(1, width - 2)
            y = random.randint(1, height - 2)
            if self.grid.get(x, y) is None:
                self.grid.set(x, y, Wall())

        # 4) Random agent placement
        self.place_agent()

        # 5) Random goal placement
        self.place_obj(Goal())

        # Mission
        self.mission = "navigate a complex maze to reach the goal"
