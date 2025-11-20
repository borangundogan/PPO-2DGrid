# src/custom_envs/easy_env.py

from __future__ import annotations

from .base_env import BaseCustomEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal


class EasyEnv(BaseCustomEnv):
    """
    Easiest environment:
    - 16x16 empty grid
    - Fixed agent start position
    - Fixed goal position
    - No internal walls, no obstacles, no doors
    """

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Add outer walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        # Fixed goal position (bottom-right)
        goal_x = width - 2
        goal_y = height - 2
        self.put_obj(Goal(), goal_x, goal_y)

        # Mission text
        self.mission = "reach the goal"
