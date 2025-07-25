import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

class QNetwork(tf.keras.Model):
    def __init__(self, input_dim, n_actions):
        super(QNetwork, self).__init__()
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.q_out = layers.Dense(n_actions, activation=None)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        return self.q_out(x)

class QAgent:
    def __init__(self, input_dim, n_actions, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.q_net = QNetwork(input_dim, n_actions)
        self.target_q_net = QNetwork(input_dim, n_actions)
        self.q_net.compile(optimizer=optimizers.Adam(lr))
        self.update_target_network()
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.n_actions = n_actions

    def update_target_network(self):
        self.target_q_net.set_weights(self.q_net.get_weights())

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        q_values = self.q_net(tf.convert_to_tensor([state], dtype=tf.float32))
        return tf.argmax(q_values[0]).numpy()

    def train(self, state, action, reward, next_state, done):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        next_state = tf.convert_to_tensor([next_state], dtype=tf.float32)

        q_next = self.target_q_net(next_state)[0]
        q_target = reward + self.gamma * tf.reduce_max(q_next) * (1 - int(done))

        with tf.GradientTape() as tape:
            q_values = self.q_net(state)
            q_action = tf.gather(q_values[0], action)
            loss = tf.reduce_mean(tf.square(q_target - q_action))
        
        grads = tape.gradient(loss, self.q_net.trainable_variables)
        self.q_net.optimizer.apply_gradients(zip(grads, self.q_net.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
