# src/custom_envs/hardest_env.py

from __future__ import annotations
import random
from .base_env import BaseCustomEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall, Goal


class HardestEnv(BaseCustomEnv):
    """
    Hardest environment:
    - FourRooms-style maze (4 quadrants)
    - Fully open connections (no doors)
    - Random agent start
    - Random goal
    - Random obstacles
    """

    def _gen_grid(self, width, height):
        # 1) Empty grid + outer walls
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        mid_x = width // 2   # = 8 for 16x16 grid
        mid_y = height // 2  # = 8

        # 2) Build FourRooms layout
        # --- vertical separator ---
        for y in range(1, height - 1):
            self.grid.set(mid_x, y, Wall())

        # --- horizontal separator ---
        for x in range(1, width - 1):
            self.grid.set(x, mid_y, Wall())

        # 3) OPEN THE PASSAGES (like classic FourRooms)
        # remove 1 cell in each wall to connect rooms

        # top-left ↔ top-right
        open_y_top = random.randint(2, mid_y - 2)
        self.grid.set(mid_x, open_y_top, None)

        # bottom-left ↔ bottom-right
        open_y_bottom = random.randint(mid_y + 2, height - 3)
        self.grid.set(mid_x, open_y_bottom, None)

        # top-left ↔ bottom-left
        open_x_left = random.randint(2, mid_x - 2)
        self.grid.set(open_x_left, mid_y, None)

        # top-right ↔ bottom-right
        open_x_right = random.randint(mid_x + 2, width - 3)
        self.grid.set(open_x_right, mid_y, None)

        # 4) Add random obstacles (6–12)
        num_obstacles = random.randint(6, 12)
        for _ in range(num_obstacles):
            x = random.randint(1, width - 2)
            y = random.randint(1, height - 2)
            if self.grid.get(x, y) is None:
                self.grid.set(x, y, Wall())

        # 5) Random agent placement
        self.place_agent()

        # 6) Random goal placement
        self.place_obj(Goal())

        self.mission = "navigate the four connected rooms to reach the goal"
