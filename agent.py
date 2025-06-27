import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow_probability as tfp
import gym
import numpy as np
from network import SACNetwork, CriticNetwork
from replay_buffer import ReplayBuffer 

tfd = tfp.distributions

class ACAgent:
    def __init__(self, min_s, max_s, input_dims, r_scale, # n_actions,
                 env=None, tau = 0.005, 
                 alpha=0.0001, gamma=0.99):   
        # Convert input_dims to integer if it's a sequence
        if isinstance(input_dims, (list, tuple)):
            input_dims = input_dims[0]

        self.alpha = alpha                    # Alpha is default learning rate
        self.gamma = gamma                    # Gamma is default discount factor
        self.tau = tau                        # Soft update parameter

        # Handle both int and tuple input_dims
        buffer_input_shape = (input_dims,) if isinstance(input_dims, int) else input_dims
        self.memory = ReplayBuffer(max_size=100000, input_shape=buffer_input_shape)
        self.batch_size = 32
        self.reward_scale = 0.01
        self.min_s = min_s
        self.max_s = max_s  

        # self.n_actions = n_actions
        self.action = None
        self.action_space = gym.spaces.Box(low=np.array([min_s]), high=np.array([max_s]), dtype=np.float32)

        self.optimizer = SGD(learning_rate=alpha)    # Initialize optimizer

        self.ac = SACNetwork(min_s, max_s) # n_actions removed
        # self.ac.build((None, input_dims))  # Explicit build call
        self.ac.compile(optimizer=SGD(learning_rate=alpha, clipnorm=1.0))

        self.critic = CriticNetwork() #min_s, max_s) 
        # self.critic.build((None, input_dims, 1))
        self.critic.compile(optimizer=SGD(learning_rate=alpha, clipnorm=1.0))

        self.target_value = SACNetwork(min_s, max_s)
        # self.target_value.build((None, input_dims))
        self.target_value.compile(optimizer=SGD(learning_rate=alpha, clipnorm=1.0))

        # Build networks explicitly
        test_state = tf.random.normal((1, input_dims))
        test_action = tf.random.normal((1, 1))
        _ = self.ac(test_state)  # This should trigger build
        _ = self.target_value(test_state)
        _ = self.critic([test_state, test_action])  # Force build

        self.target_value.set_weights(self.ac.get_weights())  # Sync initially
        self.target_weights = self.target_value.get_weights()

        self.r_scale = r_scale
        self.update_params(tau=1)  # update network parameters

    # def build(self, input_shape):
    #     super().build(input_shape)
    #     # Add optimizer AFTER build is complete
    #     self.optimizer = Adam(learning_rate=self.alpha)

    def choose_action(self, observation, evaluate=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        mean, std = self.ac.get_policy_params(state)
        
        dist = tfd.Normal(loc=mean, scale=std)
        action = mean if evaluate else dist.sample()
        
        # Ensure scalar output
        action_value = float(action.numpy().item())  # Convert to Python float
        return np.clip(action_value, self.min_s, self.max_s)
    
    def remember(self, state, action, reward, n_state, done):
        self.memory.store_transition(state, action, reward, n_state, done)

    def update_params(self, tau=None):
        if tau is None:
            tau = self.tau
        
        # Get all weights from both networks
        main_weights = self.ac.get_weights()
        target_weights = self.target_value.get_weights()
        
        # Soft update: new_target = tau * main + (1-tau) * old_target
        updated_weights = []
        for w_main, w_target in zip(main_weights, target_weights):
            updated_weight = tau * w_main + (1 - tau) * w_target
            updated_weights.append(updated_weight)
        
        # Set the updated weights
        self.target_value.set_weights(updated_weights)

    def save_model(self):
        print('Saving model......')
        self.ac.save_weights(self.ac.cp_file)
        
    def load_model(self):
        print('Loading model......')
        self.ac.load_weights(self.ac.cp_file)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample_buffer(self.batch_size)
        
        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # 1. Update Q-function (Critic)
        with tf.GradientTape() as tape:
            # Get V(next_state) from target network
            next_v, _, _ = self.target_value(next_states)  # Get value from target
            next_v = tf.squeeze(next_v)
            
            # Q target = r + γ(1-done)V(next_state)
            q_target = rewards + self.gamma * (1 - dones) * next_v
            
            # Current Q estimate
            q_pred = tf.squeeze(self.critic([states, tf.expand_dims(actions, -1)]))
            
            # MSE loss
            critic_loss = 0.5 * tf.reduce_mean((q_pred - q_target)**2)
        
        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        critic_grad = tf.clip_by_global_norm(critic_grad, 1.0)[0]  # Clip gradients
        self.critic.optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        # 2. Update Value Network
        with tf.GradientTape() as tape:
            # Sample new actions from current policy
            new_actions, log_probs = self.ac.sample_normal(states)
            
            # Get Q estimates
            q_values = tf.squeeze(self.critic([states, new_actions]))
            
            # V target = Q - α*logπ
            v_target = q_values - self.alpha * log_probs
            
            # Current V estimate
            v_pred = tf.squeeze(self.ac.get_value(states))
            
            # MSE loss
            value_loss = 0.5 * tf.reduce_mean((v_pred - tf.stop_gradient(v_target))**2)
        
        # Only update value function variables
        value_variables = (self.ac.l1.trainable_variables + 
                       self.ac.l2.trainable_variables + 
                       self.ac.value_f.trainable_variables)
        value_grad = tape.gradient(value_loss, value_variables)
        
        if value_grad and all(g is not None for g in value_grad):
            value_grad = tf.clip_by_global_norm(value_grad, 1.0)[0]
            self.ac.v_optimizer.apply_gradients(zip(value_grad, value_variables))
        else:
            print("Warning: Value gradients are None!")

        # 3. Update Policy (Actor)
        with tf.GradientTape() as tape:
            # Sample new actions
            new_actions, log_probs = self.ac.sample_normal(states)
            
            # Get Q estimates
            q_values = tf.squeeze(self.critic([states, new_actions]))
            
            # Policy objective: maximize (Q - α*logπ)
            actor_loss = tf.reduce_mean(self.alpha * log_probs - q_values)
        
        # Define policy-specific variables (exclude value function)
        policy_variables = (self.ac.l1.trainable_variables + 
                        self.ac.l2.trainable_variables + 
                        self.ac.mean.trainable_variables + 
                        self.ac.std_dev.trainable_variables)
        
        policy_grads = tape.gradient(actor_loss, policy_variables)
        
        if policy_grads and all(g is not None for g in policy_grads):
            policy_grads = tf.clip_by_global_norm(policy_grads, 1.0)[0]
            self.ac.a_optimizer.apply_gradients(zip(policy_grads, policy_variables))
        else:
            print("Warning: Policy gradients are None!")
            # Debug: print which gradients are None
            for var, grad in zip(policy_variables, policy_grads):
                if grad is None:
                    print(f"No gradient for: {var.name}")

        # 4. Update target networks
        self.update_params()

