import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Lambda

import os  # To save model checkpoints

class ACNetwork(keras.Model):
    def __init__(self, n_actions, min_s, max_s,
                 l1_dims=1024, l2_dims=512,   # No. of neurons for full connected layers 1 & 2
                 name='acn', cp_dir='acn_cp'):      # Checkpoint Directory

        super(ACNetwork, self).__init__()
        self.n_actions = n_actions
        self.name = name
        self.cp_dir = cp_dir
        self.cp_file = os.path.join(self.cp_dir, name+'_ac')
        self.min_s = min_s
        self.max_s = max_s

        self.l1_dims = l1_dims
        self.l2_dims = l2_dims
    
        self.l1 = Dense(self.l1_dims, activation='relu')
        self.l2 = Dense(self.l2_dims, activation='relu')
        self.value_f = Dense(1, activation=None)       # Value function 
        
        # Policy function, Output will be mean & standard deviation to represent distribution

        # Mean output (scaled to speed range)
        self.mean = Dense(1, activation='tanh')         # tanh gives [-1, 1]
        self.scaled_mean = Lambda(lambda x: min_s + (max_s - min_s) * (x + 1) / 2)(self.mean)  # Scale to [min, max]

        # Std Dev output (ensures positivity)
        self.std_dev = Dense(1, activation='softplus')  # Softplus ensures positive std dev


    def call(self, state):
        features = self.l2(self.l1(state))

        value = self.value_f(features)
        
        mean = self.scaled_mean(self.mean(features))
        std_dev = self.std_dev(features)

        return value, mean, std_dev




        