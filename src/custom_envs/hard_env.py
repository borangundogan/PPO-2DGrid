# src/custom_envs/hard_env.py

from __future__ import annotations
from collections import deque

from .base_env import BaseCustomEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall, Goal

class HardEnv(BaseCustomEnv):
    def _gen_grid(self, width, height):
        max_retries = 100
        self.random_goal = True
        
        for _ in range(max_retries):
            self.grid = Grid(width, height)
            self.grid.wall_rect(0, 0, width, height)

            mid = width // 2
            is_large_map = width > 10

            valid_gap_indices = list(range(1, height - 1))
            
            if is_large_map:
                num_gaps = self.np_random.integers(2, 6) 
            else:
                num_gaps = 1
                
            gap_indices = self.np_random.choice(valid_gap_indices, size=num_gaps, replace=False)

            for i in range(height):
                if i == 0 or i == height - 1:
                    continue
                
                if i not in gap_indices:
                    self.grid.set(mid, i, Wall())

            if is_large_map:
                num_extra_walls = self.np_random.integers(6, 13)
                for _ in range(num_extra_walls):
                    for _ in range(10): 
                        x = self.np_random.integers(1, width - 1)
                        y = self.np_random.integers(1, height - 1)

                        if x != mid and self.grid.get(x, y) is None:
                            self.grid.set(x, y, Wall())
                            break
            
            if not self.random_goal:
                goal_obj = self.put_obj(Goal(), width - 2, height - 2)
                goal_pos = (width - 2, height - 2)
            else:
                goal_obj = self.place_obj(Goal(), top=(mid + 1, 0), size=(width - mid - 1, height))
                if goal_obj is None:
                    continue
                goal_pos = goal_obj if isinstance(goal_obj, tuple) else goal_obj.cur_pos

            if self.agent_start_pos is not None:
                self.agent_pos = self.agent_start_pos
                self.agent_dir = self.agent_start_dir
            else:
                self.place_agent(top=(1, 1), size=(mid - 1, height - 2))

            if self._is_reachable(self.agent_pos, goal_pos, self.grid):
                self.mission = "navigate through the gaps and reach the goal"
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
                        if cell is None or isinstance(cell, Goal) or (nx == gx and ny == gy):
                            visited.add((nx, ny))
                            queue.append((nx, ny))
                            
        return False