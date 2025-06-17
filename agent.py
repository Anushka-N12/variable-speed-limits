import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import gym
import numpy as np
from network import SACNetwork
from replay_buffer import ReplayBuffer 

tfd = tfp.distributions

class ACAgent:
    def __init__(self, min_s, max_s, input_dims, r_scale, # n_actions,
                 env=None, tau = 0.005, 
                 alpha=0.0003, gamma=0.99):   # Alpha is default learning rate
        self.gamma = gamma                    # Gamma is default discount factor
        self.tau = tau                         # Tau, reward_scale?
        self.memory = ReplayBuffer(max_size=100000, input_shape=input_dims)

        # self.n_actions = n_actions
        self.action = None
        self.action_space = gym.spaces.Box(low=np.array([min_s]), high=np.array([max_s]), dtype=np.float32)

        self.ac = SACNetwork(min_s, max_s) # n_actions removed
        self.ac.compile(optimizer=Adam(learning_rate=alpha))

        self.target_value = SACNetwork(min_s, max_s)
        self.target_value.set_weights(self.ac.get_weights())  # Sync initially
        self.target_weights = self.target_value.get_weights()
        self.target_value.compile(optimizer=Adam(learning_rate=alpha))

        self.r_scale = r_scale
        self.update_params(tau=1)  # update network parameters

    def choose_action(self, observation, evaluate=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        mean, std = self.ac.get_policy_params(state)     # Call function is called automatically called 
                                                         # when model instance is created
        # Create a Gaussian distribution with reparameterization
        dist = tfd.Normal(loc=mean, scale=std)
        if evaluate:       # Use mean for evaluation (deterministic)
            action = mean
        else:              # Sample from distribution (stochastic)
            action = dist.sample()
        self.action = action
        return action.numpy()[0]
    
    def remember(self, state, action, reward, n_state, done):
        self.memory.store_transition(state, action, reward, n_state, done)

    def update_params(self, tau=None):
        if tau is None:
            tau = self.tau
        weights = []
        targets = self.target_weights
        for w_main, w_target in zip(self.ac.value_f.weights, self.target_value.value_f.weights):
            w_target.assign(tau * w_main + (1 - tau) * w_target)

        self.target_value.set_weights(weights)

    def save_model(self):
        print('Saving model......')
        self.ac.save_weights(self.ac.cp_file)
        
    def load_model(self):
        print('Loading model......')
        self.ac.load_weights(self.ac.cp_file)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        # 1. Sample a batch
        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)

        state = tf.convert_to_tensor(state, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.float32)
        done = tf.convert_to_tensor(done, dtype=tf.float32)

        # Update Value Network 
        with tf.GradientTape() as tape:
            value = tf.squeeze(self.ac.get_value(state), 1)

            # Sample from current policy
            new_action, log_prob = self.ac.sample_normal(state)
            q_value = tf.squeeze(self.ac.get_q(tf.concat([state, new_action], axis=1)), 1)

            # Target: Q - log_pi
            value_target = q_value - tf.squeeze(log_prob, 1)
            value_loss = 0.5 * tf.keras.losses.MSE(value, value_target)

        value_grads = tape.gradient(value_loss, self.ac.trainable_variables)
        self.ac.optimizer.apply_gradients(zip(value_grads, self.ac.trainable_variables))

        # Update Actor Network
        with tf.GradientTape() as tape:
            new_action, log_prob = self.ac.sample_normal(state)
            q_value = tf.squeeze(self.ac.get_q(tf.concat([state, new_action], axis=1)), 1)

            actor_loss = tf.reduce_mean(alpha * log_prob - q_value)

        actor_grads = tape.gradient(actor_loss, self.ac.trainable_variables)
        self.ac.optimizer.apply_gradients(zip(actor_grads, self.ac.trainable_variables))

        # Update Critic / Q-function
        with tf.GradientTape() as tape:
            # Compute Q target: reward + γ * V(next_state)
            value_ = tf.squeeze(self.ac.get_value(next_state), 1)
            q_target = reward + self.gamma * value_ * (1 - done)

            q_pred = tf.squeeze(self.ac.get_q(tf.concat([state, action], axis=1)), 1)
            critic_loss = 0.5 * tf.keras.losses.MSE(q_pred, q_target)

        critic_grads = tape.gradient(critic_loss, self.ac.trainable_variables)
        self.ac.optimizer.apply_gradients(zip(critic_grads, self.ac.trainable_variables))
