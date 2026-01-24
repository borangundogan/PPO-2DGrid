import gymnasium as gym

class StuckPenaltyWrapper(gym.Wrapper):
    """
    A wrapper that penalizes the agent for staying in the same position for too long.
    This catches both 'wall-banging' (trying to move forward into a wall) and 
    'spinning' (changing direction without changing coordinates).
    
    It uses a counter (max_stay) to allow for natural observation (looking around)
    before applying penalties.
    """
    def __init__(self, env, max_stay=3, penalty=-0.1):
        super().__init__(env)
        self.max_stay = max_stay  # Maximum allowed steps in the same tile
        self.penalty = penalty    # Penalty applied per step after limit is reached
        self.stay_counter = 0     # Tracks consecutive steps in the same tile
        self.last_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Reset internal state
        self.stay_counter = 0
        if hasattr(self.env.unwrapped, 'agent_pos'):
            self.last_pos = tuple(self.env.unwrapped.agent_pos)
            
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        current_pos = None
        if hasattr(self.env.unwrapped, 'agent_pos'):
            # Convert to tuple for consistent comparison
            current_pos = tuple(self.env.unwrapped.agent_pos)

        is_stuck = False

        if current_pos is not None and self.last_pos is not None:
            # Check if the agent is in the exact same coordinate as before
            if current_pos == self.last_pos:
                self.stay_counter += 1
            else:
                # Agent moved to a new tile, reset the counter
                self.stay_counter = 0
            
            # Apply penalty ONLY if the agent has stayed too long (e.g., > 3 steps)
            if self.stay_counter >= self.max_stay:
                reward += self.penalty
                is_stuck = True
            
            # Update last known position
            self.last_pos = current_pos
        
        # Log the stuck flag for analysis
        info["stuck"] = is_stuck
            
        return obs, reward, terminated, truncated, info