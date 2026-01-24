# src/custom_envs/hard_env.py

from __future__ import annotations

from .base_env import BaseCustomEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall, Door, Goal
import numpy as np

class HardEnv(BaseCustomEnv):
    """
    Hard difficulty environment with Scale-Adaptive Logic:
    
    1. 8x8 Grid:
       - Single gap in the wall.
       - No extra random walls (space is too tight).
       
    2. 16x16 Grid:
       - 2 to 3 gaps in the wall (to fix sparsity).
       - Random scattered walls added to open space (to fix featureless void).
    """

    def _gen_grid(self, width, height):
        self.random_goal = True
        self.grid = Grid(width, height)

        # 1. Generate Outer Walls
        self.grid.wall_rect(0, 0, width, height)

        mid = width // 2
        
        # --- DYNAMIC LOGIC BASED ON SIZE ---
        is_large_map = width > 10  # Threshold for 16x16 vs 8x8

        # A. Determine Gaps
        valid_gap_indices = list(range(1, height - 1))
        
        if is_large_map:
            # 16x16: 2 to 3 gaps
            num_gaps = self.np_random.integers(2, 6) 
        else:
            # 8x8: Only 1 gap (classic hard mode)
            num_gaps = 1
            
        gap_indices = self.np_random.choice(valid_gap_indices, size=num_gaps, replace=False)

        # B. Build The Middle Wall
        for i in range(height):
            if i == 0 or i == height - 1:
                continue
            
            if i not in gap_indices:
                self.grid.set(mid, i, Wall())

        # C. Add Extra Walls (ONLY for Large Maps)
        if is_large_map:
            # Add clutter to break up the empty space in 16x16
            num_extra_walls = self.np_random.integers(6, 13)

            for _ in range(num_extra_walls):
                for _ in range(10): # Try 10 times to place
                    x = self.np_random.integers(1, width - 1)
                    y = self.np_random.integers(1, height - 1)

                    # Don't block the middle line or existing objects
                    if x != mid and self.grid.get(x, y) is None:
                        self.grid.set(x, y, Wall())
                        break
        
        # 4. Place Goal
        if not self.random_goal:
            self.put_obj(Goal(), width - 2, height - 2)
        else:
            self.place_obj(Goal())

        # 5. Place Agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "navigate through the gaps and reach the goal"