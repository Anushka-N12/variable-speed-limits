from q_agent import QAgent
from sim_env import MetaNetEnv  # or whichever your environment file is
import numpy as np

env = MetaNetEnv()
agent = QAgent(input_dim=env.STATE_DIM, n_actions=7)  # actions: [60, 70, ..., 120]

n_episodes = 500
action_set = [60, 70, 80, 90, 100, 110, 120]

for ep in range(n_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action_idx = agent.choose_action(state)
        action = action_set[action_idx]
        next_state, reward, done, _ = env.step(action)
        agent.train(state, action_idx, reward, next_state, done)
        state = next_state
        total_reward += reward

    agent.update_target_network()
    print(f"Episode {ep} — Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
