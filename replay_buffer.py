'''-------------------------------------------------------------------------------
Replay Buffer:
This module implements a replay buffer for storing and sampling transitions
from the RL environment. It is used to store experiences during training
and sample batches for learning updates.

Storage includes:
- State 
- Next State
- Action
- Reward
- Done flag (terminal state)
---------------------------------------------------------------------------------'''

import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, input_shape): #, n_actions
        self.input_shape = (input_shape,) if isinstance(input_shape, int) else input_shape
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_mem = np.zeros((self.mem_size, *input_shape))
        self.n_state_mem = np.zeros((self.mem_size, *input_shape))
        self.action_mem = np.zeros((self.mem_size))  #, n_actions
        self.reward_mem = np.zeros(self.mem_size)
        self.terminal_mem = np.zeros(self.mem_size, dtype=bool)
    
    def store_transition(self, state, action, reward, n_state, done):
        assert state.shape == self.state_mem[0].shape, "Mismatch in state shape"
        index = self.mem_cntr % self.mem_size
        self.state_mem[index] = state
        self.n_state_mem[index] = n_state
        self.reward_mem[index] = reward
        self.action_mem[index] = action
        self.terminal_mem[index] = done
        self.mem_cntr += 1
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_mem[batch]
        n_states = self.n_state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        dones = self.terminal_mem[batch]
        return states, actions, rewards, n_states, dones
    
    def reset(self):
        self.mem_cntr = 0

