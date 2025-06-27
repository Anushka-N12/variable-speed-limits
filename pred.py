# Phase 2 extension:
class POMDPWrapper:
    def __init__(self, env):
        self.env = env
        self.belief_state = None
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._update_belief(obs)
        return self.belief_state, reward, done, info
        
    def _update_belief(self, obs):
        # Implement your prediction model here
        self.belief_state = np.concatenate([
            obs,
            predicted_speeds  # From your prediction model
        ])