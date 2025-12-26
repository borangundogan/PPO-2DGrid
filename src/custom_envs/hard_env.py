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
# smaller walls for easier scenario agents
    def _gen_grid(self, width, height):
        self.random_goal = True
        self.grid = Grid(width, height)

        # outer walls
        self.grid.wall_rect(0, 0, width, height)

        mid = width // 2
        doorway_y = height // 2   # middle opening

        # vertical wall with an opening
        for i in range(height):
            if i != doorway_y:
                self.grid.set(mid, i, Wall())

        # goal
        if not self.random_goal:
            self.put_obj(Goal(), width - 2, height - 2)
        else:
            self.place_obj(Goal())

        # place agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "navigate through the gap and reach the goal"