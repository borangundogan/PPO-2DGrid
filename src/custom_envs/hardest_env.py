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
        max_retries = 100
        
        for _ in range(max_retries):
            self.grid = Grid(width, height)
            self.grid.wall_rect(0, 0, width, height)

            mid_x = width // 2
            mid_y = height // 2

            for y in range(1, height - 1):
                self.grid.set(mid_x, y, Wall())
            for x in range(1, width - 1):
                self.grid.set(x, mid_y, Wall())

            open_y_top = self.np_random.integers(2, mid_y - 1)
            self.grid.set(mid_x, open_y_top, None)

            open_y_bottom = self.np_random.integers(mid_y + 1, height - 2)
            self.grid.set(mid_x, open_y_bottom, None)

            open_x_left = self.np_random.integers(2, mid_x - 1)
            self.grid.set(open_x_left, mid_y, None)

            open_x_right = self.np_random.integers(mid_x + 1, width - 2)
            self.grid.set(open_x_right, mid_y, None)

            num_obstacles = self.np_random.integers(6, 13)
            for _ in range(num_obstacles):
                x = self.np_random.integers(1, width - 1)
                y = self.np_random.integers(1, height - 1)
                if self.grid.get(x, y) is None and (x != mid_x) and (y != mid_y):
                    self.grid.set(x, y, Wall())

            self.place_agent()
            goal_obj = self.place_obj(Goal())
            
            if goal_obj is None:
                continue
                
            goal_pos = goal_obj if isinstance(goal_obj, tuple) else goal_obj.cur_pos

            if self._is_reachable(self.agent_pos, goal_pos, self.grid):
                self.mission = "navigate the four connected rooms to reach the goal"
                return

        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.place_agent()
        self.place_obj(Goal())
        self.mission = "reach the goal"

    def _is_reachable(self, start_pos, goal_pos, grid):
        from collections import deque
        sx, sy = start_pos
        gx, gy = goal_pos
        width, height = grid.width, grid.height
        
        visited = set([(sx, sy)])
        queue = deque([(sx, sy)])
        
        while queue:
            cx, cy = queue.popleft()
            if (cx, cy) == (gx, gy):
                return True 
                
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if (nx, ny) not in visited:
                        cell = grid.get(nx, ny)
                        from minigrid.core.world_object import Goal
                        if cell is None or isinstance(cell, Goal) or (nx == gx and ny == gy):
                            visited.add((nx, ny))
                            queue.append((nx, ny))
                            
        return False
