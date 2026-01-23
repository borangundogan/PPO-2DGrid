# src/custom_envs/medium_hard_env.py

from __future__ import annotations
import random
from collections import deque
import numpy as np

from .base_env import BaseCustomEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Wall

class MediumHardEnv(BaseCustomEnv):
    """
    Medium-Hard environment (Scattered Pillars):
    - Adjusts obstacle count based on grid size.
    - Guarantees a valid path from agent to goal using BFS.
    """

    def _gen_grid(self, width, height):
        max_retries = 100  # Prevent infinite loops
        
        for _ in range(max_retries):
            # 1. Create an empty grid
            self.grid = Grid(width, height)
            self.grid.wall_rect(0, 0, width, height)

            # 2. Calculate dynamic obstacle count (15-25% density)
            playable_area = (width - 2) * (height - 2)
            min_obs = int(playable_area * 0.15)
            max_obs = int(playable_area * 0.25)
            
            num_obstacles = random.randint(max(1, min_obs), max(1, max_obs))

            # 3. Place obstacles
            for _ in range(num_obstacles):
                self.place_obj(Wall(), max_tries=100)

            # 4. Place agent and goal
            self.place_agent()
            goal_obj = self.place_obj(Goal())
            
            if goal_obj is None:
                continue

            # --- FIX: Handle different return types from place_obj ---
            if isinstance(goal_obj, tuple):
                goal_pos = goal_obj
            else:
                goal_pos = goal_obj.cur_pos
            # ---------------------------------------------------------

            # 5. Check if a valid path exists
            if self._is_reachable(self.agent_pos, goal_pos, self.grid):
                self.mission = "avoid pillars and reach the goal"
                return  # Success, exit loop

        # Fallback if no valid map is found
        print("Warning: Could not generate a valid map, returning an empty map.")
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.place_agent()
        self.place_obj(Goal())
        self.mission = "reach the goal"

    def _is_reachable(self, start_pos, goal_pos, grid):
        """
        Uses Breadth-First Search (BFS) to check if the goal is reachable 
        from the start position without passing through walls.
        """
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
            
            # Check 4 directions (Right, Down, Left, Up)
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                
                if 0 <= nx < width and 0 <= ny < height:
                    if (nx, ny) not in visited:
                        cell = grid.get(nx, ny)
                        
                        # Passable if empty (None), Goal, or Agent
                        if cell is None or isinstance(cell, Goal) or (nx == gx and ny == gy):
                            visited.add((nx, ny))
                            queue.append((nx, ny))
                            
        return False