import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Lambda
import tensorflow_probabilities as tfps

import os  # To save model checkpoints

class SACNetwork(keras.Model):
    def __init__(self, min_s, max_s, # n_actions,
                 l1_dims=512, l2_dims=256,   # No. of neurons for full connected layers 1 & 2
                 name='sacn', cp_dir='sacn_cp'):      # Checkpoint Directory

        super(SACNetwork, self).__init__()
        # self.n_actions = n_actions
        self.name = name
        self.cp_dir = cp_dir
        self.cp_file = os.path.join(self.cp_dir, name+'_ac')
        self.min_s = min_s
        self.max_s = max_s

        self.l1_dims = l1_dims
        self.l2_dims = l2_dims
        self.noise = 1e-6
    
        self.l1 = Dense(self.l1_dims, activation='relu')
        self.l2 = Dense(self.l2_dims, activation='relu')
        self.value_f = Dense(1, activation=None)       # Value function 

        self.q = Dense(1, activation=None)       # Q-function 
        
        # Policy function, Output will be mean & standard deviation to represent distribution

        # Mean output (scaled to speed range)
        self.mean = Dense(1, activation='tanh')         # tanh gives [-1, 1]
        self.scaled_mean = Lambda(lambda x: min_s + (max_s - min_s) * (x + 1) / 2)(self.mean)  # Scale to [min, max]

        # Std Dev output (ensures positivity)
        self.std_dev = Dense(1, activation='softplus')  # Softplus ensures positive std dev

    def call(self, state):
        features = self.l2(self.l1(state))

        value = self.value_f(features)

        q = self.std_dev(features)
        
        mean = self.scaled_mean(self.mean(features))
        std_dev = self.std_dev(features)
        std_dev = tf.clip_by_value(std_dev, self.noise, 1)


        return value, q, mean, std_dev
    
    def sample_normal(self, state, reparameterize=True):
        mean, std_dev = self.call(state)
        probabilities = tfp.distributions.Normal(mean, std_dev)

        actions = probabilities.sample()
        action = tf.math.tanh(actions)*self.max_action
        log_probs = probabilities.log_prob(actions)
        log_probs -= tf.math.log(1-tf.math.pow(action,2)+self.noise)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)

        return action, log_probs




        