# src/custom_envs/medium_hard_env.py

from __future__ import annotations
from collections import deque
import numpy as np

from .base_env import BaseCustomEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Wall

class MediumHardEnv(BaseCustomEnv):
    def _gen_grid(self, width, height):
        max_retries = 100
        
        for _ in range(max_retries):
            self.grid = Grid(width, height)
            self.grid.wall_rect(0, 0, width, height)

            playable_area = (width - 2) * (height - 2)
            min_obs = int(playable_area * 0.15)
            max_obs = int(playable_area * 0.25)
            
            num_obstacles = self.np_random.integers(max(1, min_obs), max(1, max_obs) + 1)

            for _ in range(num_obstacles):
                self.place_obj(Wall(), max_tries=100)

            self.place_agent()
            goal_obj = self.place_obj(Goal())
            
            if goal_obj is None:
                continue

            goal_pos = goal_obj if isinstance(goal_obj, tuple) else goal_obj.cur_pos

            if self._is_reachable(self.agent_pos, goal_pos, self.grid):
                self.mission = "avoid pillars and reach the goal"
                return

        print("Warning: Could not generate a valid map, returning an empty map.")
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.place_agent()
        self.place_obj(Goal())
        self.mission = "reach the goal"

    def _is_reachable(self, start_pos, goal_pos, grid):
        sx, sy = start_pos
        gx, gy = goal_pos
        width, height = grid.width, grid.height
        
        visited = set()
        visited.add((sx, sy))
        
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
                        
                        if cell is None or isinstance(cell, Goal) or (nx == gx and ny == gy):
                            visited.add((nx, ny))
                            queue.append((nx, ny))
                            
        return False