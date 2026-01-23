import gymnasium as gym

class StuckPenaltyWrapper(gym.Wrapper):
    """
    A wrapper that penalizes the agent for staying in the same position.
    Also returns 'stuck': True in the info dict for logging purposes.
    """
    def __init__(self, env, penalty=-0.1):
        super().__init__(env)
        self.penalty = penalty
        self.last_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if hasattr(self.env.unwrapped, 'agent_pos'):
            self.last_pos = self.env.unwrapped.agent_pos
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        current_pos = None
        if hasattr(self.env.unwrapped, 'agent_pos'):
            current_pos = self.env.unwrapped.agent_pos

        # Flag to track if penalty was applied this step
        is_stuck = False

        if current_pos is not None and self.last_pos is not None:
            if current_pos == self.last_pos:
                reward += self.penalty
                is_stuck = True  # <--- LOGGING FLAG
        
        if current_pos is not None:
            self.last_pos = current_pos
            
        # Add to info dict so FOMAML can count it
        info["stuck"] = is_stuck
            
        return obs, reward, terminated, truncated, info