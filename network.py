import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Lambda, LeakyReLU
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam, SGD

import os  # To save model checkpoints

class SACNetwork(keras.Model):
    def __init__(self, min_s, max_s, # n_actions,
                 l1_dims=32, l2_dims=64,   # No. of neurons for full connected layers 1 & 2
                 name='sacn', cp_dir='sacn_cp', alpha=0.0001):      # Checkpoint Directory

        super(SACNetwork, self).__init__()
        # self.n_actions = n_actions
        self.name = name
        self.alpha = alpha
        self.cp_dir = cp_dir
        self.cp_file = os.path.join(self.cp_dir, name+'_ac'+'.weights.h5')
        self.min_s = min_s
        self.max_s = max_s
        # Input dims should match state size (2 + N*2 where N=number of sections)
        self.l1_dims = l1_dims
        self.l2_dims = l2_dims
        self.noise = 1e-6

        self.v_optimizer = SGD(learning_rate=alpha)
        self.a_optimizer = SGD(learning_rate=alpha)
    
        self.l1 = Dense(self.l1_dims)
        self.l1_activation = LeakyReLU(alpha=0.01)
        self.l2 = Dense(self.l2_dims)
        self.l2_activation = LeakyReLU(alpha=0.01)

        # Value function 
        self.value_f = Dense(1, activation=None)       

        # # Q-function / Critic
        # self.q_l1 = Dense(self.l1_dims, activation='relu')
        # self.q_l2 = Dense(self.l2_dims, activation='relu')
        # self.q_f = Dense(1, activation=None)       
        
        # Policy function, Output will be mean & standard deviation to represent distribution

        # Mean output (scaled to speed range)
        self.mean = Dense(1, activation='tanh')         # tanh gives [-1, 1]
        self.scaled_mean = Lambda(lambda x: min_s + (max_s - min_s) * (x + 1) / 2)  # Scale to [min, max]

        # Std Dev output (ensures positivity)
        self.std_dev = Dense(1, activation='softplus')  # Softplus ensures positive std dev
        # Build network
        # self.build((None, 14))

    def call(self, state):
        ex = self.l1_activation(self.l1(state))
        features = self.l2_activation(self.l2(ex))

        value = self.value_f(features)

        # # Critic value
        # q = self.q_f(features)
        
        mean = self.scaled_mean(self.mean(features))
        std_dev = self.std_dev(features)
        std_dev = tf.clip_by_value(std_dev, self.noise, 1)

        return value, mean, std_dev #q,
    
    def get_value(self, state):
        ex = self.l1_activation(self.l1(state))
        features = self.l2_activation(self.l2(ex))
        return self.value_f(features)

    # def get_q(self, state_action):
    #     ex = self.q_l1(state_action)
    #     features = self.q_l2(ex)
    #     return self.q(features)

    def get_policy_params(self, state):
        ex = self.l1_activation(self.l1(state))
        features = self.l2_activation(self.l2(ex))
        mean = self.scaled_mean(self.mean(features))
        std_dev = tf.clip_by_value(self.std_dev(features), self.noise, 1.0)
        return mean, std_dev
    
    def sample_normal(self, state, reparameterize=True):
        _, mean, std_dev = self.call(state)
        probabilities = tfp.distributions.Normal(mean, std_dev)

        actions = probabilities.sample()
        action = tf.math.tanh(actions) * self.max_s
        log_probs = probabilities.log_prob(actions)
        # log_probs -= tf.math.log(1-tf.math.pow(action,2)+self.noise)
        # Make sure the argument to log is always positive:
        log_arg = tf.clip_by_value(1-tf.math.pow(action,2)+self.noise, self.noise, 1.0)
        log_probs -= tf.math.log(log_arg)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)

        return action, log_probs


class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=16, fc2_dims=32, name='critic', cp_dir='tmp', alpha=0.0001):
        super(CriticNetwork, self).__init__()
        self.cp_file = os.path.join(cp_dir, name+'_critic')

        # Initialize optimizer
        self.optimizer = SGD(learning_rate=alpha)

        self.fc1 = Dense(fc1_dims)
        self.fc1_activation = LeakyReLU(alpha=0.01)
        self.fc2 = Dense(fc2_dims)
        self.fc2_activation = LeakyReLU(alpha=0.01)
        self.q = Dense(1, activation=None)  # Q-value output

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, state_action):
        state, action = state_action
        # Concatenate state and action
        x = tf.concat([state, action], axis=-1)
        x = self.fc1_activation(self.fc1(x))
        x = self.fc2_activation(self.fc2(x))
        return self.q(x)