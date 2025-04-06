import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import gym
import numpy as np
from network import ACNetwork

tfd = tfp.distributions

class ACAgent:
    def __init__(self,  n_actions, min_s, max_s,
                 alpha=0.0003, gamma=0.99):   # Alpha is default learning rate
        self.gamma = gamma                    # Gamma is default discount factor
        self.n_actions = n_actions
        self.action = None
        self.action_space = gym.spaces.Box(low=np.array([min_s]), high=np.array([max_s]), dtype=np.float32)

        self.ac = ACNetwork(n_actions, min_s, max_s)
        self.ac.compile(optimizer=Adam(learning_rate=alpha))

    def choose_action(self, observation, evaluate=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        std, mean = self.ac(state)     # Call function is called automatically called 
                                       # when model instance is created
        # Create a Gaussian distribution with reparameterization
        dist = tfd.Normal(loc=mean, scale=std)
        if evaluate:       # Use mean for evaluation (deterministic)
            action = mean
        else:              # Sample from distribution (stochastic)
            action = dist.sample()
        self.action = action
        return action.numpy()[0]

    def save_models(self):
        print('Saving models......')
        self.ac.save_weights(self.ac.cp_file)
        
    def load_models(self):
        print('Loading models......')
        self.ac.load_weights(self.ac.cp_file)

    def learn(self, state, reward, n_state, done):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        n_state = tf.convert_to_tensor([n_state], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        with tf.GradientTape(persistent = True) as tape:
            state_value, mean, std = self.ac(state)
            n_state_value, _, _ = self.ac(n_state)
            state_value = tf.squeeze(state_value)
            n_state_value = tf.squeeze(n_state_value)
            
            dist = tfd.Normal(loc=mean, scale=std)
            log_prob = dist.log_prob(self.action)
            delta = reward + self.gamma*n_state_value(1-int(done)) - state_value
            a_loss = log_prob*delta
            c_loss = delta**2
            t_loss = a_loss + c_loss
            gradient = tape.gradient(t_loss, self.ac.trainable_variables)
            self.ac.optimizer.apply_gradients(zip(gradient, self.ac.trainable_variables))