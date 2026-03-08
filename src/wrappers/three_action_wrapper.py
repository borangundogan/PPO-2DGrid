import gymnasium as gym
import numpy as np

class ThreeActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(3)
        base_actions = env.unwrapped.actions
        
        self._action_map = np.array([
            base_actions.left,
            base_actions.right,
            base_actions.forward
        ], dtype=np.int64)

    def action(self, act):
        return self._action_map[act]