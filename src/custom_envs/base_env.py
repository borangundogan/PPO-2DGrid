# src/custom_envs/base_env.py

from __future__ import annotations

from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace


class BaseCustomEnv(MiniGridEnv):
    """
    Base class for all MERLIN custom environments.
    Subclass this and override _gen_grid() to define your own world layout.
    """

    def __init__(
        self,
        size: int = 16,
        agent_start_pos=None,
        agent_start_dir: int = 0,
        max_steps: int | None = None,
        **kwargs,
    ):
        # Store agent configs
        self.size = size
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        # Mission: PPO does not use it, but MiniGrid requires a string output
        mission_space = MissionSpace(mission_func=self._gen_mission)

        # Default number of steps: approx. 4 * grid_size^2 (MiniGrid standard)
        if max_steps is None:
            max_steps = 4 * (size ** 2)

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=True,  # Enables efficient rendering
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "reach the goal"

    # ----------------------------------------------------------------------
    # IMPORTANT:
    # Subclasses must override this function to generate the grid layout.
    # width/height == size
    # ----------------------------------------------------------------------
    def _gen_grid(self, width, height):
        raise NotImplementedError(
            "Subclasses must implement _gen_grid() to define environment layout."
        )
