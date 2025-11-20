import gymnasium as gym

class ThreeActionWrapper(gym.ActionWrapper):
    """
    Reduce MiniGrid's 7-action space to 3 simple navigation actions:
    0 → left
    1 → right
    2 → forward
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(3)

        # IMPORTANT: use unwrapped env to access internal MiniGrid actions
        base_env = env.unwrapped

        self._left = base_env.actions.left
        self._right = base_env.actions.right
        self._forward = base_env.actions.forward

    def action(self, act):
        if act == 0:
            return self._left
        if act == 1:
            return self._right
        return self._forward
